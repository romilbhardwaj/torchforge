# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SkyPilot JobTrait implementation for TorchForge using JobGroups.

SkyPilotJob allows running Monarch workers on Kubernetes and cloud VMs via SkyPilot.
This enables TorchForge to provision and manage distributed training workloads
across Kubernetes clusters and cloud VMs (AWS, GCP, Azure, etc.).

This implementation uses SkyPilot JobGroups to launch heterogeneous resources:
- Each mesh (generator, trainer, replay_buffer, etc.) is a separate Task
- Each Task can have different resource requirements (GPUs, CPUs, memory)
- All Tasks run in parallel and can communicate via network

Requirements:
    - pip install torchmonarch-nightly (or torchmonarch)
    - pip install skypilot[kubernetes] (or other cloud backends)

Adapted from the Monarch SkyPilot integration:
https://github.com/pytorch/monarch/tree/main/examples/skypilot
"""

import logging
import os
import tempfile
import time
from typing import TYPE_CHECKING, Any

from monarch._src.job.job import JobState, JobTrait

# If running inside a SkyPilot cluster, unset the in-cluster context variable
# to allow launching new clusters on the same Kubernetes cluster.
# This must be done before importing sky to affect the API server.
if "SKYPILOT_IN_CLUSTER_CONTEXT_NAME" in os.environ:
    del os.environ["SKYPILOT_IN_CLUSTER_CONTEXT_NAME"]

if TYPE_CHECKING:
    import sky

try:
    import sky
    import sky.jobs as sky_jobs
    from sky import global_user_state

    HAS_SKYPILOT = True
except ImportError:
    HAS_SKYPILOT = False
    sky = None  # type: ignore[assignment]
    sky_jobs = None  # type: ignore[assignment]
    global_user_state = None  # type: ignore[assignment]


logger: logging.Logger = logging.getLogger(__name__)


def _sanitize_dns_name(name: str) -> str:
    """Sanitize a name to be DNS-compatible (replace underscores with hyphens)."""
    return name.replace("_", "-")


# Default port for Monarch TCP communication
MONARCH_WORKER_PORT = 22222

# Timeout for waiting for the job to reach RUNNING status.
JOB_TIMEOUT = 900  # seconds - increased for TorchForge setup with model download

# Default setup commands to install TorchForge and Monarch on remote workers.
# Requires a Docker image with Ubuntu 22.04+ with CUDA and Python 3.10+.
# For faster cold starts, use a custom Docker image with dependencies pre-installed.
# NOTE: The version must match the driver's Monarch version to avoid serialization issues.
DEFAULT_MONARCH_VERSION = "2025.12.17"

# Key dependency versions - must match driver environment
# vllm comes from the forge preview index (specified in pyproject.toml [tool.uv.sources])
# We use uv to install TorchForge WITH deps so it respects the uv.sources configuration
#
# NOTE: These setup commands are smart about detecting pre-installed dependencies.
# With the pre-built romilb/torchforge:latest image, most steps are skipped.
# With the base pytorch image, full installation is performed.
DEFAULT_SETUP_COMMANDS = f"""
set -ex

# Check if TorchForge and dependencies are already installed (pre-built image)
DEPS_INSTALLED=false
if python -c "import forge; import monarch; import vllm" 2>/dev/null; then
    echo "TorchForge, Monarch, and vLLM already installed (using pre-built image)"
    DEPS_INSTALLED=true
    python -c "import forge; import monarch; import vllm; print(f'forge OK, monarch OK, vllm={{vllm.__version__}}')"
fi

# Install dependencies if not already present
if [ "$DEPS_INSTALLED" = "false" ]; then
    echo "Dependencies not found, installing from scratch..."
    
    # Install git (required for pip installing packages with git dependencies)
    apt-get update && apt-get install -y git
    
    # Install TorchForge WITH all dependencies using uv (respects pyproject.toml [tool.uv.sources])
    # This ensures vllm comes from the correct index (preview/forge)
    if [ -f ~/sky_workdir/pyproject.toml ]; then
        echo "Installing TorchForge and all dependencies from synced workdir..."
        cd ~/sky_workdir
        
        # Install TorchForge with all deps - uv will use [tool.uv.sources] to get vllm from forge index
        uv pip install --system -e .
        
        # Verify vllm version
        python -c "import vllm; print(f'Installed vllm version: {{vllm.__version__}}')"
        
        # Pin specific versions that must match driver
        uv pip install --system "torchmonarch-nightly=={DEFAULT_MONARCH_VERSION}"
        uv pip install --system "transformers>=4.50,<5.0"
    fi
fi

# Pre-download HuggingFace models if MODEL_NAME env var is set
# This is needed because TorchTitan's checkpointer expects models to be local
# Use --local-dir to download directly to the expected path (e.g., Qwen/Qwen3-8B)
echo "MODEL_NAME env var is: '$MODEL_NAME'"
if [ -n "$MODEL_NAME" ]; then
    echo "Pre-downloading model: $MODEL_NAME to local directory"
    export HF_HUB_ENABLE_HF_TRANSFER=1
    cd ~/sky_workdir
    huggingface-cli download "$MODEL_NAME" --local-dir "$MODEL_NAME"
    echo "Model downloaded to ~/sky_workdir/$MODEL_NAME"
    ls -la "$MODEL_NAME" | head -10
else
    echo "MODEL_NAME is not set, skipping model download"
fi

echo "Done with TorchForge worker setup"
"""
# Pre-built TorchForge image with all dependencies installed
# This dramatically speeds up worker initialization (minutes instead of 10+ minutes)
# To rebuild: see experimental/skypilot/docker-builder.sky.yaml
DEFAULT_IMAGE_ID = "docker:romilb/torchforge:latest"


def _configure_transport() -> None:
    """Configure the Monarch transport using the public API."""
    from monarch.actor import enable_transport

    enable_transport("tcp")


def _attach_to_workers_wrapper(name: str, ca: str, workers: list[str]):
    """Wrapper around attach_to_workers with deferred import."""
    from monarch._src.actor.bootstrap import attach_to_workers

    return attach_to_workers(name=name, ca=ca, workers=workers)


class SkyPilotJob(JobTrait):
    """
    SkyPilotJob to provision and manage Monarch workers on K8s and cloud VMs.

    Uses SkyPilot JobGroups to launch heterogeneous resources - each mesh
    (generator, trainer, replay_buffer, etc.) is a separate Task with its
    own resource requirements.

    Example:
        >>> from forge.controller.skypilot_job import SkyPilotJob
        >>>
        >>> job = SkyPilotJob(
        ...     meshes={"trainer": 2, "generator": 1, "replay_buffer": 1},
        ...     default_mesh_resources={"accelerators": "H100:1", "cpus": "4+"},
        ...     mesh_resources={
        ...         "generator": {"accelerators": "H100:2", "memory": "64+"},
        ...         "replay_buffer": {"accelerators": None, "cpus": "8+"},  # CPU-only
        ...     },
        ...     cloud="kubernetes",
        ... )
        >>> state = job.state()
        >>> trainers = state.trainer  # HostMesh with 2 nodes
        >>> generator = state.generator  # HostMesh with 1 node
    """

    def __init__(
        self,
        meshes: dict[str, int],
        default_mesh_resources: dict[str, Any] | None = None,
        mesh_resources: dict[str, dict[str, Any]] | None = None,
        cloud: str | None = None,
        image_id: str | None = None,
        cluster_name: str | None = None,
        monarch_port: int = MONARCH_WORKER_PORT,
        idle_minutes_to_autostop: int | None = None,
        python_exe: str = "python",
        setup_commands: str | None = None,
        workdir: str | None = None,
        file_mounts: dict[str, str] | None = None,
        envs: dict[str, str] | None = None,
    ) -> None:
        """
        Args:
            meshes: Dictionary mapping mesh names to number of nodes.
                    e.g., {"trainer": 2, "generator": 1}
            default_mesh_resources: Default resources for all meshes.
                    e.g., {"accelerators": "H100:1", "cpus": "4+", "memory": "32+"}
            mesh_resources: Per-mesh resource overrides (merged with defaults).
                    e.g., {"generator": {"accelerators": "H100:2"}}
            cloud: Cloud provider (kubernetes, aws, gcp, azure).
            image_id: Docker image for workers.
            cluster_name: Base name for the SkyPilot clusters.
            monarch_port: Port for Monarch worker communication.
            idle_minutes_to_autostop: Auto-cleanup timeout.
            python_exe: Python executable to use.
            setup_commands: Custom setup commands (defaults to TorchForge install).
            workdir: Local directory to sync to workers.
            file_mounts: Additional file mounts.
            envs: Environment variables for workers.
        """
        if not HAS_SKYPILOT:
            raise ImportError(
                "SkyPilot is not installed. Install it with: pip install skypilot[kubernetes]"
            )

        # Configure transport at runtime when Monarch is available
        try:
            _configure_transport()
        except ImportError:
            # Monarch bindings not available, will fail later when needed
            pass

        super().__init__()

        self._meshes = meshes
        self._default_mesh_resources = default_mesh_resources or {}
        self._mesh_resources = mesh_resources or {}
        self._cloud = cloud
        self._image_id = image_id or DEFAULT_IMAGE_ID
        self._cluster_name = cluster_name
        self._port = monarch_port
        self._idle_minutes_to_autostop = idle_minutes_to_autostop
        self._python_exe = python_exe
        self._setup_commands = setup_commands
        self._workdir = workdir
        self._file_mounts = file_mounts
        self._envs = envs or {}

        # Runtime state
        self._job_id: int | None = None
        self._mesh_ips: dict[str, list[str]] = {}  # mesh_name -> list of IPs

    def _get_mesh_resources(self, mesh_name: str) -> dict[str, Any]:
        """Get resources for a mesh, merging defaults with overrides."""
        resources = dict(self._default_mesh_resources)
        if mesh_name in self._mesh_resources:
            resources.update(self._mesh_resources[mesh_name])
        return resources

    def _build_sky_resources(self, mesh_name: str) -> "sky.Resources":
        """Build sky.Resources for a specific mesh."""
        res = self._get_mesh_resources(mesh_name)
        kwargs = {}

        # Set cloud
        if self._cloud:
            cloud_map = {
                "kubernetes": sky.Kubernetes,
                "aws": sky.AWS,
                "gcp": sky.GCP,
                "azure": sky.Azure,
            }
            cloud_cls = cloud_map.get(self._cloud.lower())
            if cloud_cls:
                kwargs["cloud"] = cloud_cls()

        # Set accelerators (GPUs)
        accelerators = res.get("accelerators")
        if accelerators and accelerators != "null":
            kwargs["accelerators"] = accelerators

        # Set CPUs
        cpus = res.get("cpus")
        if cpus:
            kwargs["cpus"] = cpus

        # Set memory
        memory = res.get("memory")
        if memory:
            kwargs["memory"] = memory

        # Set image - per-mesh override or global default
        image_id = res.get("image_id", self._image_id)
        if image_id:
            kwargs["image_id"] = image_id

        return sky.Resources(**kwargs)

    def _build_worker_command(self) -> str:
        """Build the bash command to start Monarch workers on each node."""
        python_code = f'''
import socket
import logging
import sys

# Enable verbose logging
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

hostname = socket.gethostname()
ip_addr = socket.gethostbyname(hostname)
address = f"tcp://{{ip_addr}}:{self._port}"
print(f"Starting Monarch worker at {{address}} (hostname={{hostname}})", flush=True)
sys.stdout.flush()

try:
    from monarch.actor import run_worker_loop_forever
    print(f"Imported run_worker_loop_forever successfully", flush=True)
    print(f"Worker ready and listening...", flush=True)
    run_worker_loop_forever(address=address, ca="trust_all_connections")
except Exception as e:
    print(f"ERROR in worker: {{e}}", flush=True)
    import traceback
    traceback.print_exc()
    raise
'''
        escaped_code = python_code.replace("'", "'\"'\"'")
        env_vars = " ".join(
            [
                f"export HYPERACTOR_HOST_SPAWN_READY_TIMEOUT={JOB_TIMEOUT}s",
                f"export HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT={JOB_TIMEOUT}s",
                f"export HYPERACTOR_MESH_PROC_SPAWN_MAX_IDLE={JOB_TIMEOUT}s",
            ]
        )
        return f"{env_vars} && {self._python_exe} -c '{escaped_code}'"

    def _mesh_has_gpus(self, mesh_name: str) -> bool:
        """Check if a mesh has GPU accelerators configured."""
        resources = self._get_mesh_resources(mesh_name)
        accelerators = resources.get("accelerators")
        return accelerators is not None and accelerators != "null"

    def _create_job_group_yaml(self) -> str:
        """Create a multi-document YAML string for the JobGroup."""
        import yaml

        # Header document - sanitize for DNS compatibility
        job_group_name = _sanitize_dns_name(self._cluster_name or f"forge-{os.getpid()}")
        header = {"name": job_group_name, "execution": "parallel"}

        # Use provided setup commands or default
        setup = (
            self._setup_commands
            if self._setup_commands is not None
            else DEFAULT_SETUP_COMMANDS
        )

        # Worker command
        run_cmd = self._build_worker_command()

        # Create task documents for each mesh
        task_docs = []
        for mesh_name, num_nodes in self._meshes.items():
            resources = self._build_sky_resources(mesh_name)

            # Convert resources to YAML-compatible dict
            res_dict = {}
            if resources.cloud:
                res_dict["cloud"] = str(resources.cloud).lower()
            if resources.accelerators:
                # Convert accelerators dict to string format
                for acc_type, acc_count in resources.accelerators.items():
                    res_dict["accelerators"] = f"{acc_type}:{acc_count}"
                    break
            if resources.cpus:
                res_dict["cpus"] = resources.cpus
            if resources.memory:
                res_dict["memory"] = resources.memory
            if resources.image_id:
                res_dict["image_id"] = resources.image_id

            # Sanitize mesh name for DNS compatibility
            dns_safe_name = _sanitize_dns_name(mesh_name)
            task_doc = {
                "name": dns_safe_name,
                "resources": res_dict,
                "num_nodes": num_nodes,
                "setup": setup,
                "run": run_cmd,
            }

            # Add workdir if specified
            if self._workdir:
                task_doc["workdir"] = self._workdir

            # Add file_mounts if specified
            if self._file_mounts:
                task_doc["file_mounts"] = self._file_mounts

            # Add envs - only include MODEL_NAME for GPU meshes (CPU meshes don't need model)
            task_envs = {}
            if self._envs:
                for key, value in self._envs.items():
                    # Only pass MODEL_NAME to GPU meshes
                    if key == "MODEL_NAME" and not self._mesh_has_gpus(mesh_name):
                        logger.info(f"Skipping MODEL_NAME for CPU-only mesh '{mesh_name}'")
                        continue
                    task_envs[key] = value
            if task_envs:
                task_doc["envs"] = task_envs

            task_docs.append(task_doc)

        # Combine into multi-document YAML
        yaml_parts = ["---", yaml.dump(header, default_flow_style=False)]
        for task_doc in task_docs:
            yaml_parts.append("---")
            yaml_parts.append(yaml.dump(task_doc, default_flow_style=False))

        return "\n".join(yaml_parts)

    def _create(self, client_script: str | None) -> None:
        """Launch a SkyPilot JobGroup with tasks for each mesh."""
        if client_script is not None:
            raise RuntimeError("SkyPilotJob cannot run batch-mode scripts yet")

        # Create JobGroup YAML
        yaml_content = self._create_job_group_yaml()
        logger.info(f"Creating JobGroup with YAML:\n{yaml_content}")

        # Write YAML to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            # Load as DAG
            from sky.utils import dag_utils

            dag = dag_utils.load_dag_from_yaml(yaml_path)

            # Launch the JobGroup
            logger.info(f"Launching JobGroup with {len(self._meshes)} tasks...")
            request_id = sky_jobs.launch(dag)
            result = sky.get(request_id)

            # Result is (job_id, handle)
            if isinstance(result, tuple):
                self._job_id = result[0]
            else:
                self._job_id = result

            logger.info(f"JobGroup launched with job_id: {self._job_id}")

        except Exception as e:
            logger.error(f"Failed to launch JobGroup: {e}")
            self._kill()
            raise RuntimeError(f"Failed to launch JobGroup: {e}") from e
        finally:
            # Clean up temp file
            try:
                os.unlink(yaml_path)
            except Exception:
                pass

        # Wait for all tasks to be RUNNING
        try:
            self._wait_for_job_group_running(timeout=JOB_TIMEOUT)
        except Exception as e:
            logger.error(f"JobGroup failed to reach RUNNING status: {e}")
            self._kill()
            raise

        logger.info("All tasks RUNNING, workers will start after setup completes")

    def _wait_for_job_group_running(self, timeout: int = JOB_TIMEOUT) -> None:
        """Wait for all tasks in the JobGroup to reach RUNNING status.
        
        Uses two strategies:
        1. Try SkyPilot jobs queue API (may fail in consolidation mode)
        2. Fall back to checking cluster handles directly
        """
        start_time = time.time()
        poll_interval = 10  # seconds
        # Use sanitized names to match what we created in the YAML
        expected_tasks = set(_sanitize_dns_name(name) for name in self._meshes.keys())
        api_failures = 0

        logger.info(
            f"Waiting for JobGroup tasks to start (timeout={timeout}s)..."
        )

        while time.time() - start_time < timeout:
            running_tasks = set()
            failed_tasks = set()
            
            try:
                # Strategy 1: Try jobs queue API
                request_id = sky_jobs.queue(
                    refresh=False, job_ids=[self._job_id]
                )
                jobs = sky.get(request_id)

                for task_record in jobs:
                    task_name = task_record.get("task_name")
                    if task_name not in expected_tasks:
                        continue

                    status = str(task_record.get("status", ""))

                    if "RUNNING" in status:
                        running_tasks.add(task_name)
                    elif "FAILED" in status or "CANCELLED" in status:
                        failed_tasks.add(task_name)

                if failed_tasks:
                    raise RuntimeError(
                        f"Tasks failed: {failed_tasks}. "
                        f"Check logs with: sky jobs logs {self._job_id}"
                    )
                
                api_failures = 0  # Reset on success

            except Exception as e:
                api_failures += 1
                # Strategy 2: Fall back to checking cluster handles
                if api_failures >= 3:
                    logger.info("Job queue API not working, checking cluster handles...")
                    for mesh_name in self._meshes.keys():
                        dns_safe_name = _sanitize_dns_name(mesh_name)
                        cluster_name = self._generate_cluster_name(dns_safe_name, self._job_id)
                        try:
                            handle = global_user_state.get_handle_from_cluster_name(cluster_name)
                            if handle is not None and handle.stable_internal_external_ips:
                                running_tasks.add(dns_safe_name)
                        except Exception:
                            pass
                else:
                    logger.warning(f"Error checking job status: {e}")

            # Check if all tasks are running
            if running_tasks == expected_tasks:
                logger.info(f"All {len(expected_tasks)} tasks are RUNNING")
                return

            elapsed = int(time.time() - start_time)
            logger.info(
                f"Tasks running: {len(running_tasks)}/{len(expected_tasks)} "
                f"(waited {elapsed}s)"
            )

            time.sleep(poll_interval)

        raise RuntimeError(
            f"Timeout waiting for JobGroup tasks to reach RUNNING status"
        )

    def _generate_cluster_name(self, task_name: str, job_id: int) -> str:
        """Generate managed job cluster name using SkyPilot's naming convention."""
        # Match SkyPilot's generate_managed_job_cluster_name logic
        # Truncate task name to 30 chars and append job_id
        from sky.utils import common_utils
        from sky.jobs import constants as jobs_constants

        cluster_name = common_utils.make_cluster_name_on_cloud(
            task_name,
            jobs_constants.JOBS_CLUSTER_NAME_PREFIX_LENGTH,
            add_user_hash=False,
        )
        return f"{cluster_name}-{job_id}"

    def _get_mesh_ips(self, mesh_name: str) -> list[str]:
        """Get IP addresses for a specific mesh's cluster.
        
        Uses SkyPilot's cluster handle to get the actual pod/VM IPs.
        These IPs are routable from the driver pod to the worker pods.
        """
        # Sanitize mesh name for DNS compatibility (matching what we used in YAML)
        dns_safe_name = _sanitize_dns_name(mesh_name)
        # Generate cluster name using SkyPilot's naming convention
        cluster_name = self._generate_cluster_name(dns_safe_name, self._job_id)
        logger.info(f"Looking for cluster '{cluster_name}' for mesh '{mesh_name}'")

        # Get handle from cluster name
        handle = global_user_state.get_handle_from_cluster_name(cluster_name)
        if handle is None:
            raise RuntimeError(
                f"No handle found for cluster '{cluster_name}'. "
                f"Cluster may not be fully initialized yet."
            )

        if handle.stable_internal_external_ips is None:
            raise RuntimeError(f"Cluster '{cluster_name}' has no IP information")

        # Extract IPs - prefer internal for K8s, external for cloud
        ips = []
        for internal_ip, external_ip in handle.stable_internal_external_ips:
            # For K8s, use internal IP (pod IP)
            ip = internal_ip if internal_ip else external_ip
            if ip:
                ips.append(ip)

        logger.info(f"Mesh '{mesh_name}' IPs: {ips}")
        return ips

    def _state(self) -> JobState:
        """Get the current state with HostMesh objects for each mesh."""
        logger.info("Getting job state (this will block until pods/VMs are provisioned)...")
        
        if not self._jobs_active():
            # This can happen if the API server doesn't track jobs properly
            # in consolidation mode, but the pods may still be running
            logger.warning(
                "Job status check returned not active, but will try to get IPs anyway"
            )
        else:
            logger.info("Job is running, returning current state")

        # Get IPs for each mesh
        host_meshes = {}

        for mesh_name, num_nodes in self._meshes.items():
            # Get IPs if not cached
            if mesh_name not in self._mesh_ips:
                ips = self._get_mesh_ips(mesh_name)
                if len(ips) < num_nodes:
                    raise RuntimeError(
                        f"Expected {num_nodes} nodes for '{mesh_name}', "
                        f"got {len(ips)}"
                    )
                self._mesh_ips[mesh_name] = ips[:num_nodes]

            mesh_ips = self._mesh_ips[mesh_name]
            workers = [f"tcp://{ip}:{self._port}" for ip in mesh_ips]
            logger.info(f"Connecting to workers for mesh '{mesh_name}': {workers}")

            host_mesh = _attach_to_workers_wrapper(
                name=mesh_name,
                ca="trust_all_connections",
                workers=workers,
            )

            # Wait for the host mesh to be initialized
            logger.info(f"Waiting for host mesh '{mesh_name}' to initialize...")
            host_mesh.initialized.get()
            logger.info(f"Host mesh '{mesh_name}' ready")

            host_meshes[mesh_name] = host_mesh

        return JobState(host_meshes)

    def can_run(self, spec: "JobTrait") -> bool:
        """Check if this job can run the given spec."""
        if not isinstance(spec, SkyPilotJob):
            return False

        return (
            spec._meshes == self._meshes
            and spec._default_mesh_resources == self._default_mesh_resources
            and spec._mesh_resources == self._mesh_resources
            and spec._port == self._port
            and self._jobs_active()
        )

    def _jobs_active(self) -> bool:
        """Check if the SkyPilot JobGroup is still active.
        
        In consolidation mode, the jobs controller runs locally and the API
        may have issues tracking job state. We're lenient here because the
        actual pods may be running even if the API doesn't report them.
        """
        if not self.active or self._job_id is None:
            return False

        try:
            # Don't use refresh=True to avoid jobs controller issues
            # in consolidation mode. Just check cached status.
            request_id = sky_jobs.queue(refresh=False, job_ids=[self._job_id])
            jobs = sky.get(request_id)

            # Check if any tasks are still running
            for task_record in jobs:
                status = str(task_record.get("status", ""))
                if "RUNNING" in status:
                    return True

            # If we got results but none are running, jobs may have completed/failed
            if jobs:
                return False
            
            # Empty result - could be API issue, assume jobs are still active
            logger.debug(f"Empty job queue result for job {self._job_id}, assuming active")
            return True

        except Exception as e:
            error_msg = str(e).lower()
            # Known errors where jobs might still be running:
            # - "does not exist" + "controller" - consolidation mode issue
            # - "no in-progress managed jobs" - API server not tracking jobs
            # In both cases, the pods may still be running
            if any(pattern in error_msg for pattern in [
                "does not exist",
                "no in-progress managed jobs",
                "clusternotuperror",
            ]):
                logger.debug(
                    f"Job status check failed ({e}), assuming job {self._job_id} is still active"
                )
                return True
            logger.warning(f"Error checking job status: {e}")
            return False

    def _kill(self) -> None:
        """Cancel the SkyPilot JobGroup."""
        if self._job_id is not None:
            try:
                logger.info(f"Cancelling SkyPilot JobGroup {self._job_id}")
                request_id = sky_jobs.cancel(job_ids=[self._job_id])
                sky.get(request_id)
                logger.info(f"JobGroup {self._job_id} cancelled")
            except Exception as e:
                logger.warning(f"Failed to cancel job: {e}")

        self._job_id = None
        self._mesh_ips.clear()
