# Build this with 
# docker build -t vggt-test .

# Run docker container with
# docker run -it --gpus all vggt-test /bin/bash

# Then when ready to build and push to remote
# docker build -t gcr.io/tour-project-442218/vggt . && docker push gcr.io/tour-project-442218/vggt
# Then to run it
# docker run -it --gpus all gcr.io/tour-project-442218/vggt /bin/bash

# Once inside run
# gcloud storage rsync -r gs://tour_storage/data/tandt data/tandt
# TODO: Fill out the needed command
# gcloud storage rsync -r data/tandt/truck gs://tour_storage/data/tandt/truck

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Install git
RUN apt-get update && apt-get install -y curl gnupg && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update && apt-get install -y \
    google-cloud-cli \
    git \
    wget \
    unzip \
    cmake \
    build-essential \
    ninja-build \
    libglew-dev \
    libassimp-dev \
    libboost-all-dev \
    libgtk-3-dev \
    libopencv-dev \
    libglfw3-dev \
    libavdevice-dev \
    libavcodec-dev \
    libeigen3-dev \
    libxxf86vm-dev \
    tmux \
    && rm -rf /var/lib/apt/lists/*
