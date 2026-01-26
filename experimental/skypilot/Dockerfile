# TorchForge Docker Image
# 
# This image contains all dependencies for TorchForge GRPO training,
# pre-installed for fast startup. Can be used for both driver and worker pods.
#
# Build: docker build -t <registry>/torchforge:latest .
# Push: docker push <registry>/torchforge:latest

FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    socat \
    curl \
    git \
    openssh-client \
    rsync \
    && rm -rf /var/lib/apt/lists/*

# Install kubectl
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && chmod +x kubectl \
    && mv kubectl /usr/local/bin/

# Install uv for fast Python package management (already in base image, but ensure it's there)
RUN pip install --no-cache-dir uv

# Create workdir
WORKDIR /app

# Copy TorchForge source code
COPY . /app/

# Install TorchForge and all dependencies
# Using uv with --system to install into system Python
RUN uv pip install --system -e ".[dev]" \
    && uv pip install --system "transformers>=4.50,<5.0"

# Install SkyPilot from source (master branch for JobGroups support)
# Use regular install (not editable) since editable installs don't work well in Docker
RUN cd /opt \
    && git clone https://github.com/skypilot-org/skypilot.git \
    && cd skypilot \
    && uv pip install --system ".[kubernetes]"

# Verify installations
RUN python -c "import forge; print(f'TorchForge installed')" \
    && python -c "import monarch; print(f'Monarch installed')" \
    && python -c "import vllm; print(f'vLLM version: {vllm.__version__}')" \
    && python -c "import sky; print(f'SkyPilot version: {sky.__version__}')"

# Set default workdir for runtime
WORKDIR /root/sky_workdir

# Default command (can be overridden)
CMD ["bash"]
