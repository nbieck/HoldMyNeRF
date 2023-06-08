FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Create volumes to persist model checkpoints. This is for documentation: use the -v tag to actually mount the volumes in docker run.
VOLUME /app/model /root/.u2net/

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
    # For downloading SEEM checkpoint:
    wget \
    # Required for video to image, and rendering video:
    ffmpeg \
    # Cleanup:
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*rm 


# The entire app is installed inside /app
WORKDIR /app

# Copy the repo
COPY . .

# Assign environment variables
RUN bash env_file.sh

# Update and initialize submodules
RUN git submodule update --init --recursive

# Build instant-ngp. If you get error 137 (insufficient memory), lower the '-j' parameter
WORKDIR /app/dependencies/instant_ngp
RUN cmake . -B build
RUN cmake --build build --config RelWithDebInfo -j 4

# Build COLMAP
ENV CC=/usr/bin/gcc-10
ENV CXX=/usr/bin/g++-10
ENV CUDAHOSTCXX=/usr/bin/g++-10
WORKDIR /app/dependencies/colmap
RUN cmake --version && cmake . -B build -GNinja -D CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} && \
    ninja -C build -j 4 && \
    ninja -C build install

# Return to app directory
WORKDIR /app

# Install Python requirements
RUN pip3 install --no-cache-dir -r requirements/linux/requirements.txt && \
    pip3 install --no-cache-dir -r requirements/linux/requirements_git.txt 


# Setup for Gradio
EXPOSE 7860

# Launch the Gradio app on localhost:7860
CMD ["python3", "app.py", "--server_name", "0.0.0.0"]
