FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /app

# Update apt-get and install packages
RUN apt-get update && apt-get install -y --no-install-recommends \ 
    # Line endings fix:
    dos2unix \
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
    # For downloading SEEM checkpoint:
    wget \
    # Cleanup:
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*rm 

# Copy and install Python requirements
COPY requirements /app/
RUN pip3 install --no-cache-dir \ 
    -r ./requirements/linux/requirements.txt \
    -r ./requirements/linux/requirements_git.txt 

# Copy the repo
COPY . /app/

# Update and initialize submodules
RUN git submodule update --init --recursive

# Build instant-ngp
WORKDIR /app/dependencies/instant_ngp
RUN cmake . -B build
# If you get error 137 (insufficient memory), lower the '-j 4' parameter
RUN cmake --build build --config RelWithDebInfo -j 4

# Setup for Gradio
EXPOSE 7860

# Launch the Gradio app on localhost:7860
WORKDIR /app
CMD ["python3", "app.py", "--server_name", "0.0.0.0"]
