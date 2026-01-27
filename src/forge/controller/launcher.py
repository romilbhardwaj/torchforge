# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Launcher specific logic (i.e. SLURM, k8s when supported, etc.)"""

import atexit
import logging

from forge.controller.base import BaseLauncher
from forge.types import Launcher, LauncherConfig
from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
from monarch._rust_bindings.monarch_hyperactor.config import configure
from monarch.actor import ProcMesh
from monarch.job import JobState, JobTrait, SlurmJob

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


JOB_NAME_KEY = "job_name"
LAUNCHER_KEY = "launcher"


def get_meshes_from_config(cfg: LauncherConfig) -> dict[str, int]:
    """Extract mesh requirements from launcher config.

    Args:
        cfg: The launcher configuration

    Returns:
        Dictionary mapping mesh names to number of hosts required
    """
    meshes: dict[str, int] = {}

    # Add services that need remote hosts
    for service_name, service_cfg in cfg.services.items():
        hosts = getattr(service_cfg, "hosts", None)
        if hosts and hosts > 0:
            mesh_name = service_cfg.mesh_name or service_name
            meshes[mesh_name] = hosts

    # Add actors that need remote hosts
    for actor_name, actor_cfg in cfg.actors.items():
        hosts = getattr(actor_cfg, "hosts", None)
        if hosts and hosts > 0:
            mesh_name = actor_cfg.mesh_name or actor_name
            meshes[mesh_name] = hosts

    return meshes


class Slurmlauncher(BaseLauncher):
    def __init__(
        self,
        cfg: LauncherConfig,
    ):
        self.cfg = cfg

    async def initialize(self) -> tuple[JobTrait, JobState]:
        """Initialize the launcher and create a single SlurmJob for all resources.

        This pre-allocates all meshes defined in the config in one Slurm job.

        Returns:
            A tuple of (job, job_state) containing the SlurmJob handle and its state.
        """
        # HostMesh currently requires explicit configuration
        # of the underlying transport from client to mesh.
        # This can be removed in the future once this has been removed.
        configure(default_transport=ChannelTransport.TcpWithHostname)

        # Collect all mesh requirements from config
        meshes = get_meshes_from_config(self.cfg)

        # If no remote resources needed, skip job creation
        if not meshes:
            return

        # Build slurm_args from config
        slurm_args = [f"--{key}={value}" for key, value in self.cfg.slurm_args.items()]

        # Create a single SlurmJob with all meshes
        logger.info(f"Creating SlurmJob with meshes: {meshes}")
        job = SlurmJob(
            meshes=meshes,  # e.g., {"generator": 1, "trainer": 2, "ref_model": 1}
            slurm_args=slurm_args,
            job_name=self.cfg.job_name + "_workers" or "forge_job",
            time_limit="72:00:00",  # Default to 72 hours
            gpus_per_node=self.cfg.gpus_per_node,
            cpus_per_task=self.cfg.cpus_per_task,
            mem=self.cfg.mem,
        )

        # Apply the job to allocate resources
        logger.info("Submitting SlurmJob...")
        job.apply()
        logger.info("SlurmJob submitted, waiting for allocation...")

        # Register cleanup handler
        atexit.register(job.kill)

        # Wait for job allocation
        logger.info("Getting job state (this will block until nodes are allocated)...")
        job_state = job.state(cached_path=None)

        logger.info("SlurmLauncher initialization complete.")
        return job, job_state

    async def remote_setup(self, procs: ProcMesh) -> None:
        return


class SkyPilotLauncher(BaseLauncher):
    """Launcher for running TorchForge on Kubernetes and cloud VMs via SkyPilot.

    This launcher provisions cloud instances or Kubernetes pods using SkyPilot
    JobGroups, allowing heterogeneous resources per mesh. Each mesh (generator,
    trainer, replay_buffer, etc.) is launched as a separate Task with its own
    resource requirements.

    The driver must be running inside the Kubernetes cluster for K8s backends.

    Example config:
        provisioner:
          launcher: skypilot
          job_name: my_grpo_job
          skypilot_args:
            cloud: kubernetes
            image_id: docker:pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
            idle_minutes_to_autostop: 30
            model_name: Qwen/Qwen3-8B
            default_mesh_resources:
              accelerators: "H100:1"
              cpus: "4+"
              memory: "32+"
            mesh_resources:
              generator:
                accelerators: "H100:2"
              replay_buffer:
                accelerators: null
                cpus: "8+"
    """

    def __init__(self, cfg: LauncherConfig):
        self.cfg = cfg

    async def initialize(self) -> tuple[JobTrait, JobState]:
        """Initialize the SkyPilot launcher and provision cloud resources.

        Returns:
            A tuple of (job, job_state) containing the SkyPilotJob handle and its state.
        """
        # Import SkyPilotJob here to avoid import errors when SkyPilot is not installed
        try:
            from forge.controller.skypilot_job import SkyPilotJob
        except ImportError as err:
            raise ImportError(
                "SkyPilot is not installed. Install it with: "
                "pip install skypilot[kubernetes]"
            ) from err

        # Collect all mesh requirements from config
        meshes = get_meshes_from_config(self.cfg)

        # If no remote resources needed, skip job creation
        if not meshes:
            return None, None

        # Parse skypilot_args
        skypilot_args = self.cfg.skypilot_args or {}

        cloud = skypilot_args.get("cloud")
        infra = skypilot_args.get("infra")  # e.g., "kubernetes/sky-dev"
        image_id = skypilot_args.get("image_id")
        idle_minutes_to_autostop = skypilot_args.get("idle_minutes_to_autostop", 30)
        model_name = skypilot_args.get("model_name")
        default_mesh_resources = skypilot_args.get("default_mesh_resources", {})
        mesh_resources = skypilot_args.get("mesh_resources", {})

        # Create SkyPilotJob with JobGroups support
        logger.info(f"Creating SkyPilotJob with meshes: {meshes}")
        logger.info(f"Default mesh resources: {default_mesh_resources}")
        logger.info(f"Per-mesh resources: {mesh_resources}")

        # Find TorchForge project root (contains pyproject.toml)
        import pathlib

        import forge

        forge_root = pathlib.Path(forge.__file__).parent.parent.parent
        workdir = str(forge_root) if (forge_root / "pyproject.toml").exists() else None

        # Prepare environment variables for workers
        worker_envs = {}
        if model_name:
            worker_envs["MODEL_NAME"] = model_name
            logger.info(f"Setting MODEL_NAME env var for workers: {model_name}")
        else:
            logger.warning("model_name not set in skypilot_args, workers won't pre-download model")

        job = SkyPilotJob(
            meshes=meshes,
            default_mesh_resources=default_mesh_resources,
            mesh_resources=mesh_resources,
            cloud=cloud,
            infra=infra,  # e.g., "kubernetes/sky-dev" - ensures workers go to same cluster as driver
            image_id=image_id,
            cluster_name=self.cfg.job_name + "_workers" if self.cfg.job_name else None,
            idle_minutes_to_autostop=idle_minutes_to_autostop,
            workdir=workdir,  # Sync TorchForge to workers for actor serialization
            envs=worker_envs if worker_envs else None,
        )

        # Apply the job to allocate resources
        logger.info("Launching SkyPilot JobGroup...")
        job.apply()

        # Register cleanup handler
        atexit.register(job.kill)

        # Wait for job allocation and get state
        logger.info(
            "Getting job state (this will block until pods/VMs are provisioned)..."
        )
        job_state = job.state(cached_path=None)

        logger.info("SkyPilotLauncher initialization complete.")
        return job, job_state

    async def remote_setup(self, procs: ProcMesh) -> None:
        return


def get_launcher(cfg: LauncherConfig | None = None) -> BaseLauncher | None:
    if not cfg:
        return None
    if cfg.launcher == Launcher.SLURM:
        return Slurmlauncher(cfg)
    elif cfg.launcher == Launcher.MAST:
        try:
            from forge.fb.mast_launcher import MastLauncher

            return MastLauncher(cfg, detached=False)
        except ImportError as err:
            raise ValueError("MAST is not available, cannot launch MAST jobs.") from err
    elif cfg.launcher == Launcher.SKYPILOT:
        try:
            # SkyPilot import is handled inside SkyPilotLauncher.initialize()
            return SkyPilotLauncher(cfg)
        except ImportError as err:
            raise ValueError(
                "SkyPilot is not installed. Install it with: "
                "pip install skypilot[kubernetes]"
            ) from err
    else:
        raise ValueError(f"Unsupported config provided, got {cfg}")
