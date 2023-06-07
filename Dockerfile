FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

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
    # Cleanup:
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*rm 

# Clone the repo
RUN git clone https://github.com/nbieck/HoldMyNeRF.git --recurse-submodules app

# Build instant-ngp
WORKDIR /app/dependencies/instant_ngp
RUN cmake . -B build
# If you get error 137 (insufficient memory), lower the '-j 4' parameter
RUN cmake --build build --config RelWithDebInfo -j 4
WORKDIR /app

# Install all python requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Install SEEM requirements
RUN pip3 install git+https://github.com/arogozhnikov/einops.git
RUN pip3 install git+https://github.com/MaureenZOU/detectron2-xyz.git
RUN pip3 install git+https://github.com/openai/whisper.git

# Expose port 7860 for the Gradio app
EXPOSE 7860

# Launch the Gradio app
CMD ["python3", "app.py"]
