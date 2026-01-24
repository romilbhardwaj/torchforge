# Running TorchForge on Kubernetes and Cloud VMs via SkyPilot

This directory contains examples for running TorchForge GRPO training on **Kubernetes and cloud VMs** via [SkyPilot](https://github.com/skypilot-org/skypilot).

## Overview

The SkyPilot integration allows TorchForge to provision and manage distributed training workloads across:
- **Kubernetes** (any cluster)
- **Hyperscalers**: AWS, GCP, Azure
- **Neoclouds**: CoreWeave, Nebius, and [20+ other clouds](https://docs.skypilot.co/en/latest/getting-started/installation.html)

### Architecture

The integration uses **SkyPilot JobGroups** to launch heterogeneous resources - each mesh (generator, trainer, ref_model) is a separate Task with its own resource requirements.

```mermaid
flowchart LR
    subgraph Local["🖥️ Local Machine"]
        Laptop["$ sky launch"]
    end

    subgraph K8s["☸️ Kubernetes Cluster"]
        subgraph Driver["Driver Pod"]
            Controller["GRPO Controller"]
            Provisioner["Provisioner"]
            Launcher["SkyPilotLauncher"]
        end

        subgraph JobGroup["SkyPilot JobGroup"]
            Generator["Task: generator<br/>(H100:2, 64GB)"]
            Trainer["Task: trainer<br/>(H100:1)"]
            RefModel["Task: ref_model<br/>(H100:1)"]
        end
    end

    Laptop -->|"sky launch"| Driver
    Launcher -->|"jobs.launch"| JobGroup
    Launcher <-->|"TCP:22222"| Generator
    Launcher <-->|"TCP:22222"| Trainer
    Launcher <-->|"TCP:22222"| RefModel
```

**How it works:**
1. You run `sky launch` from your laptop to start the driver pod
2. The driver runs `apps.grpo.main` with `launcher: skypilot` in the config
3. `SkyPilotLauncher` creates a **JobGroup** with separate Tasks for each mesh
4. Each Task can have different resources (GPU types, CPU, memory)
5. Workers install TorchForge and download HuggingFace models during setup
6. The driver connects to Monarch workers over TCP (port 22222)
7. Actors (Generator, Trainer, RefModel) are spawned on their respective Task pods

## Quickstart

### Prerequisites

1. **Install SkyPilot nightly** on your local machine:

> **Note**: The JobGroups feature requires `skypilot-nightly`, not the stable release.

```bash
pip install skypilot-nightly[kubernetes]  # For Kubernetes
pip install skypilot-nightly[aws]         # For AWS
pip install skypilot-nightly[gcp]         # For GCP
pip install skypilot-nightly[all]         # For all clouds
```

2. **Verify SkyPilot setup**:

```bash
sky check
sky show-gpus --infra kubernetes  # For K8s
```

For detailed setup, see the [SkyPilot documentation](https://docs.skypilot.co/en/latest/getting-started/installation.html).

### Running GRPO Training

From your local machine (with kubectl/kubeconfig configured):

```bash
cd torchforge/experimental/skypilot
sky launch torchforge_grpo.sky.yaml -c forge-grpo
```

This will:
1. Launch a driver pod in your Kubernetes cluster
2. Install TorchForge and dependencies on the driver
3. Start the GRPO training loop
4. Launch a **JobGroup** with separate Tasks for Generator, Trainer, and RefModel
5. Each Task installs TorchForge and downloads models
6. Begin training with metrics logging

### Monitoring and Debugging

```bash
# View driver logs
sky logs forge-grpo

# SSH into the driver pod
ssh forge-grpo

# View JobGroup status (from driver pod)
ssh forge-grpo "sky jobs queue"

# View worker logs
ssh forge-grpo "sky jobs logs <job_id>"
```

### Cleanup

```bash
# Tear down the driver (workers are cleaned up automatically via JobGroup)
sky down forge-grpo

# Or cancel just the JobGroup workers
ssh forge-grpo "sky jobs cancel <job_id>"
```

## Configuration Reference

### SkyPilot Args (skypilot_args)

The SkyPilot integration is configured via the `skypilot_args` dictionary in the provisioner section. This is similar to `slurm_args` for Slurm.

```yaml
provisioner:
  launcher: skypilot
  job_name: my_grpo_job
  skypilot_args:
    # Cloud configuration
    cloud: kubernetes      # Cloud provider: kubernetes, aws, gcp, azure
    image_id: docker:pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
    idle_minutes_to_autostop: 30
    model_name: Qwen/Qwen3-8B  # HuggingFace model to pre-download

    # Default resources for all meshes
    default_mesh_resources:
      accelerators: "H100:1"  # GPU spec
      cpus: "4+"              # Minimum CPUs
      memory: "32+"           # Minimum memory (GB)

    # Per-mesh resource overrides (merged with defaults)
    mesh_resources:
      generator:
        accelerators: "H100:2"  # Generator needs more GPUs
        memory: "64+"
      trainer:
        accelerators: "H100:1"
      ref_model:
        accelerators: "H100:1"
      # CPU-only meshes can override accelerators to null
      # replay_buffer:
      #   accelerators: null
      #   cpus: "8+"
```

### Resource Specification

Resources are specified per-mesh and merged with defaults:

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `accelerators` | str | GPU type and count | `"H100:2"`, `"A100:4"` |
| `cpus` | str | CPU requirement | `"4+"`, `"8"` |
| `memory` | str | Memory in GB | `"32+"`, `"64"` |
| `image_id` | str | Docker image (per-mesh override) | `"docker:my-image:tag"` |

**Resolution order**: `mesh_resources[mesh_name]` → `default_mesh_resources` → SkyPilot defaults

### Model Pre-download

The `model_name` parameter is **required** for HuggingFace models. Workers will download the model during setup using:

```bash
huggingface-cli download <model_name> --local-dir <model_name>
```

This ensures TorchTitan's checkpointer can find the model at the expected path (e.g., `Qwen/Qwen3-8B`).

> **Note**: When using `hf://` prefixes in your config (e.g., `hf://Qwen/Qwen3-8B`), the SkyPilot launcher automatically strips the prefix. Workers download models themselves rather than relying on the driver's cache.

### Resource Allocations

Services and actors with `hosts: N` will be provisioned on remote worker pods:

```yaml
services:
  generator:
    procs: 1              # Number of processes (typically matches tensor_parallel_size)
    hosts: 1              # Runs on 1 remote worker pod
    with_gpus: true
    mesh_name: generator

actors:
  trainer:
    procs: 1
    hosts: 1              # Runs on 1 remote worker pod
    with_gpus: true
    mesh_name: trainer
```

**GPU Assignment**: Each actor/service with `with_gpus: true` gets `procs` GPUs assigned. The `accelerators` config (e.g., `H100:1`) specifies how many GPUs are available per worker node. Ensure `procs` ≤ GPUs per node.

Services/actors without `hosts` or with `hosts: 0` run locally on the driver pod.

## Heterogeneous Resources Example

The power of JobGroups is running different GPU types and CPU-only workers:

```yaml
provisioner:
  launcher: skypilot
  job_name: heterogeneous_example
  skypilot_args:
    cloud: aws
    model_name: Qwen/Qwen3-8B
    
    default_mesh_resources:
      cpus: "4+"
      memory: "16+"
    
    mesh_resources:
      # GPU workers with different GPU types
      generator:
        accelerators: "A100:2"
        memory: "128+"
      trainer:
        accelerators: "H100:1"
      ref_model:
        accelerators: "A10G:1"  # Cheaper GPU for reference model
      
      # CPU-only workers (no accelerators)
      replay_buffer:
        accelerators: null
        cpus: "16+"
        memory: "256+"
      reward_actor:
        accelerators: null
        cpus: "8+"
```

## Supported Clouds

| Cloud | Installation | Notes |
|-------|--------------|-------|
| Kubernetes | `pip install skypilot[kubernetes]` | Driver must run inside cluster |
| AWS | `pip install skypilot[aws]` | Requires AWS credentials |
| GCP | `pip install skypilot[gcp]` | Requires GCP credentials |
| Azure | `pip install skypilot[azure]` | Requires Azure credentials |

See [SkyPilot Cloud Setup](https://docs.skypilot.co/en/latest/getting-started/installation.html) for credential configuration.

## Network Requirements

- **Kubernetes**: The driver pod must be inside the same cluster as worker pods
- **Cloud VMs**: Security groups must allow inbound traffic on port 22222

## Dependency Management

Worker pods install dependencies using `uv pip install -e .` from the synced TorchForge repository. This respects `pyproject.toml`'s `[tool.uv.sources]` section, which is critical for:

- **vllm**: Must come from the PyTorch forge preview index (contains development version with required modules)
- **torch**: Comes from the cu128 index for CUDA 12.8 support

Key version requirements:
- `torchmonarch-nightly`: Must match the driver's version exactly (serialization compatibility)
- `transformers>=4.50,<5.0`: Required for vllm compatibility

## Performance Tips

### Reduce Workdir Sync Time

Add a `.skyignore` file in the TorchForge root to exclude large directories:

```
# .skyignore
.git/
*.pyc
__pycache__/
.venv/
```

This significantly speeds up the workdir sync, especially if your `.git` directory is large.

### Optimize Setup Time

For faster cold starts, consider using a custom Docker image with pre-installed dependencies:

```yaml
skypilot_args:
  image_id: docker:your-registry/torchforge-base:latest
```

## Troubleshooting

### Check SkyPilot Configuration

```bash
sky check
sky show-gpus --infra kubernetes
```

### View Worker Logs

Worker pods are managed by the JobGroup. To view logs:

```bash
# From your laptop
ssh forge-grpo  # SSH into driver

# From driver, view JobGroup status
sky jobs queue

# View logs for a specific job
sky jobs logs <job_id>
sky jobs logs <job_id> --controller  # Controller logs
```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: No module named 'vllm.executor'` | Wrong vllm version on workers | Ensure workers install TorchForge using `uv pip install -e .` (respects pyproject.toml index sources) |
| `checkpoint.initial_load_path is not valid` | Model not downloaded on workers | Add `model_name` to `skypilot_args` |
| `SkyPilot is not installed` | Missing SkyPilot on driver | Check `torchforge_grpo.sky.yaml` setup script installs `skypilot[kubernetes]` |
| Connection timeout | Network issues | Ensure driver and workers are in the same cluster/VPC; check port 22222 is open |
| GPU not available | Scheduling issues | Check `sky show-gpus --infra kubernetes` for available GPUs |
| Pod scheduling issues | Resource constraints | Check Kubernetes node resources, taints, and tolerations |
| `Timeout spawning proc mesh` | Mismatch between requested procs and available GPUs | Ensure `procs` ≤ `accelerators` GPU count (e.g., `procs: 1` with `H100:1`) |
| Slow workdir sync | Large .git directory | Add `.skyignore` with `.git/` |
| `torchmonarch` serialization error | Version mismatch | Ensure workers install the exact same `torchmonarch-nightly` version as driver |

### Debugging Workflow

1. **Launch and monitor**:
   ```bash
   sky launch torchforge_grpo.sky.yaml -c forge-grpo
   sky logs forge-grpo -f  # Follow logs
   ```

2. **If training fails, check JobGroup**:
   ```bash
   ssh forge-grpo
   sky jobs queue  # Find job ID
   sky jobs logs <job_id>  # View task logs
   ```

3. **Check versions on workers**:
   ```bash
   # Get worker cluster name from jobs queue output
   ssh <worker-cluster-name>
   python -c "import vllm; print(vllm.__version__)"
   python -c "import monarch; print(monarch.__version__)"
   ls -la Qwen/Qwen3-8B  # Verify model downloaded
   ```

4. **Iterate on driver** (without restarting workers):
   ```bash
   # Edit config locally, then rsync to driver
   rsync -avz src/forge/ forge-grpo:~/sky_workdir/src/forge/
   ssh forge-grpo "cd ~/sky_workdir && python -m apps.grpo.main --config ..."
   ```

## Files in This Directory

| File | Description |
|------|-------------|
| `torchforge_grpo.sky.yaml` | SkyPilot task YAML to launch the driver pod |
| `qwen3_8b.yaml` | TorchForge GRPO config for Qwen3-8B on SkyPilot |
| `README.md` | This documentation |

## Implementation Details

The SkyPilot integration is implemented in:

- `src/forge/controller/launcher.py`: `SkyPilotLauncher` class that interfaces with SkyPilot
- `src/forge/controller/skypilot_job.py`: `SkyPilotJob` Monarch JobTrait using JobGroups
- `src/forge/types.py`: `LauncherConfig` with `skypilot_args` dict
- `src/forge/util/config.py`: Logic to strip `hf://` prefixes for SkyPilot

The `SkyPilotJob` creates a **multi-document YAML** for the JobGroup:
- First document: JobGroup header with `execution: parallel`
- Subsequent documents: One Task per mesh with its resources

Worker setup script in `skypilot_job.py`:
1. Installs git and system dependencies
2. Installs TorchForge with all dependencies via `uv pip install -e .`
3. Pins `torchmonarch-nightly` and `transformers` versions
4. Downloads HuggingFace models using `huggingface-cli download --local-dir`
