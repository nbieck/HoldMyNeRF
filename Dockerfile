FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Build arguments. Change these according to Troubleshooting in README.md.
ARG CUDA_ARCHITECTURES=75
ARG NUM_JOBS=4

# Create volumes to persist model checkpoints. This is for documentation: use the -v tag to actually mount the volumes in docker run.
VOLUME /app/model /root/.u2net/

# Prevent stop building ubuntu at time zone selection (from COLMAP Dockerfile).  
ENV DEBIAN_FRONTEND=noninteractive

# Update apt-get and install packages
RUN apt-get update && apt-get install -y --no-install-recommends \ 
    # instant-ngp requirements:
    gcc \
    cmake \
    build-essential \
    git \
    python3-dev \
    python3-pip \
    libopenexr-dev \
    libxi-dev \
    libglfw3-dev \
    libglew-dev \
    libomp-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxrandr-dev \
    # COLMAP requirements:
    # gcc-10 and g++-10 required for ubuntu22.04
    gcc-10 g++-10 \
    ninja-build \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libsqlite3-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    nvidia-container-toolkit \
    # For downloading SEEM checkpoint:
    wget \
    # Required for video to image, and rendering video:
    ffmpeg \
    # Cleanup:
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*rm 

# The entire app is installed inside /app
WORKDIR /app
COPY .gitmodules /app/
ADD .git /app/.git

# Update and initialize submodules
RUN git submodule update --init --recursive

# Build instant-ngp. If you get error 137 (insufficient memory), lower the '-j' parameter
WORKDIR /app/dependencies/instant_ngp
RUN cmake . -B build && \
    cmake --build build --config RelWithDebInfo -j ${NUM_JOBS}

# Build COLMAP
WORKDIR /app/dependencies/colmap
ENV QT_XCB_GL_INTEGRATION=xcb_egl \
    CC=/usr/bin/gcc-10 \
    CXX=/usr/bin/g++-10 \
    CUDAHOSTCXX=/usr/bin/g++-10
RUN mkdir build && \
    cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
    ninja -j ${NUM_JOBS} && \
    ninja install && \
    cd .. && rm -rf colmap

# Return to app directory
WORKDIR /app

# Install Python requirements
ADD requirements /app/requirements
RUN pip3 install --no-cache-dir -r requirements/linux/requirements.txt && \
    pip3 install --no-cache-dir -r requirements/linux/requirements_git.txt 

# Copy the repo
COPY . .

# Setup for Gradio
EXPOSE 7860

# Launch the Gradio app on localhost:7860
CMD ["python3", "app.py", "--server_name", "0.0.0.0"]
