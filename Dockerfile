# Use NVIDIA's official TensorFlow image
FROM nvcr.io/nvidia/tensorflow:24.10-tf2-py3

# Install system dependencies required for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    librosa \
    python-osc \
    sounddevice \
    tqdm \
    matplotlib

# Set the working directory in the container
WORKDIR /app

# Copy the local project files into the container
COPY . /app

# Set a default command to verify python version (can be overridden)
CMD ["python", "--version"]