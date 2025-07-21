ARG UBUNTU_VER=24.04
ARG CUDA_VER=12.6.0

FROM nvidia/cuda:${CUDA_VER}-devel-ubuntu${UBUNTU_VER}

# Build arguments. Change these according to Troubleshooting in README.md.
ARG CUDA_ARCHITECTURES=86
ARG NUM_JOBS=4

# Create volumes to persist model checkpoints. This is for documentation: use the -v tag to actually mount the volumes in docker run.
VOLUME /app/model /root/.u2net/

# Prevent stop building ubuntu at time zone selection (from COLMAP Dockerfile).  
ENV DEBIAN_FRONTEND=noninteractive

# Update apt-get and install packages
RUN apt-get update && apt-get install -y --no-install-recommends \ 
    # instant-ngp requirements:
    gcc \
    clang \
    cmake \
    build-essential \
    git \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-full \
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

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONHOME=

RUN python3 -m pip install --upgrade pip setuptools wheel

# The entire app is installed inside /app
WORKDIR /app
COPY .gitmodules /app/
ADD .git /app/.git

# Install Python requirements
WORKDIR /app
ADD requirements /app/requirements
RUN pip3 install --no-cache-dir -r requirements/linux/requirements.txt && \
    pip3 install --no-cache-dir -r requirements/linux/requirements_git.txt 

# Update and initialize submodules
RUN git submodule update --init --recursive

# Build instant-ngp. If you get error 137 (insufficient memory), lower the '-j' parameter
WORKDIR /app/dependencies/instant_ngp
ENV CC=/usr/bin/clang \
    CXX=/usr/bin/clang++ \
    CUDAHOSTCXX=/usr/bin/clang++
RUN cmake . -B build -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
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
    cd ..

#build glomap
WORKDIR /app/dependencies
RUN wget https://github.com/colmap/glomap/archive/refs/tags/1.1.0.tar.gz
RUN tar -xf 1.1.0.tar.gz
WORKDIR /app/dependencies/glomap-1.1.0
RUN mkdir build && \
        cd build && \
        cmake .. -GNinja && \
        ninja && ninja install

# init hloc
WORKDIR /app/dependencies/hloc
RUN python3 -m pip install -e .
WORKDIR /app

# Copy the repo
COPY . .

# Setup for Gradio
EXPOSE 7860

# Launch the Gradio app on localhost:7860
CMD ["python3", "app.py", "--server_name", "0.0.0.0"]
