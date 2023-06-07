FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

### === INSTALLS === ###

# Update apt so that new packages can be installed properly. wget for gazebo, dos2unix for line endings fix
RUN apt-get update && apt-get install -y --no-install-recommends \ 
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*rm 

# Clone the repo
RUN git clone https://github.com/nbieck/HoldMyNeRF.git --recurse-submodules app

# ENV CUDAToolkit_ROOT=""

# Build instant-ngp
WORKDIR /app/dependencies/instant_ngp
RUN cmake . -B build
RUN cmake --build build --config RelWithDebInfo -j
WORKDIR /app

# Install all python requirements
COPY req.txt /app
RUN pip3 install --no-cache-dir -r req.txt

COPY main.py /app

# Set the PATH environment variable
# ENV PATH="/usr/bin/python3:${PATH}"


# Install SEEM requirements
# RUN pip3 install git+https://github.com/arogozhnikov/einops.git
# RUN pip3 install git+https://github.com/MaureenZOU/detectron2-xyz.git
# RUN pip3 install git+https://github.com/openai/whisper.git


### === CLEANUP === ###

# Use dos2unix to convert the line endings, remove dos2unix, then clean up files created by apt-get
# RUN dos2unix /entrypoint.sh && apt-get --purge remove -y dos2unix && rm -rf /var/lib/apt/lists/*

# Required in Linux systems. Gives proper permissions to entrypoint file.
# RUN ["chmod", "+x", "/entrypoint.sh"]

EXPOSE 7860

CMD ["python3", "main.py"]
