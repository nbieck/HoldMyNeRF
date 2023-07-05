# :beers: Hold My NeRF!


[Niklas Bieck](https://github.com/nbieck) | [Fabiano Junior Maia Manschein](https://github.com/Fabulani) | [Yuya Takagi](https://github.com/shiohiyoko)

__Jean Monnet University (UJM)__

__Imaging and Light in Extended Reality (IMLEX)__

__[Code](https://github.com/nbieck/HoldMyNeRF)&nbsp;__


[Neural Radiance Fields (NeRFs)](https://www.matthewtancik.com/nerf) is a method that achieves state-of-the-art results for synthesizing novel views of complex scenes by optimizing an underlying continuous volumetric scene function using a sparse set of input views. In other words, it enables the 3D representation of a scene captured from multiple images of different angles/positions.

In this project, we have introduced a novel approach to capturing scenes, specifically by utilizing a static camera position and rotating the scene itself. Traditionally, scene captures involve moving the camera around the scene. However, we have devised a method that enables the representation of the scene as a NeRF by eliminating the background and other unrelated objects from the scene. By doing so, we create the illusion of the camera actively moving around the scene, despite its stationary position. This innovative technique offers a fresh perspective on scene capture and expands the possibilities for creating dynamic and immersive visual experiences.

**Hold My NeRF!** was developed as the final project in the *Complex Computer Rendering Methods in Real Time* course in [Jean Monnet University (UJM)](https://www.univ-st-etienne.fr/en/index.html). This course is a part of the [Erasmus Mundus Japan - Master of Science in Imaging and Light in Extended Reality (IMLEX)](https://imlex.org) program.


__Table of contents:__
- [Requirements](#requirements)
- [Installation](#installation)
  - [Docker](#docker)
- [Hold your own NeRF](#hold-your-own-nerf)
  - [Recording a dataset](#recording-a-dataset)
  - [Launching the app](#launching-the-app)
  - [Using the UI](#using-the-ui)
- [Examples](#examples)
- [Future work](#future-work)
- [Notes](#notes)
- [Troubleshooting](#troubleshooting)
  - [Insufficient CUDA memory and/or unsupported GPU architecture when building Docker](#insufficient-cuda-memory-andor-unsupported-gpu-architecture-when-building-docker)
- [Acknowledgements](#acknowledgements)


## Requirements

For a local installation:
- [instant-ngp requirements](https://github.com/NVlabs/instant-ngp#requirements)
- [FFmpeg](https://www.ffmpeg.org/)
- [colmap requirements](https://colmap.github.io/install.html#build-from-source) (only mandatory on Linux)

For installation with Docker containers:
- [Docker](https://www.docker.com)
  
## Installation

**NOTE:** If you prefer to use Docker, go to the [Docker](#docker) section.

Clone the repo and initialize submodules with

```sh
git clone https://github.com/nbieck/HoldMyNeRF.git --recurse-submodules
```

If you just cloned the repository, but didn't initialize the submodules, run the following:

```sh
git submodule update --init --recursive
```

Install Python requirements:

```sh
# Windows
./requirements_windows.bat

# Linux
./requirements_linux.sh
```

Now, you'll need to build `instant-ngp`. Follow the [instructions in the instant-ngp repository](https://github.com/NVlabs/instant-ngp#building-instant-ngp-windows--linux).

**IMPORTANT** when building `instant-ngp`, ensure that you have the same python environment active as where the requirements were installed.

If on **Linux**, build `colmap`. Follow the [instructions](https://colmap.github.io/install.html#linux).

On **Windows**, building `colmap` is **optional**, as a compiled binary will automatically be downloaded if necessary. If you want to build it yourself (in order to benefit from GPU
acceleration, for example), see the [instructions](https://colmap.github.io/install.html#id3).

Finally, start the app on [localhost:7860](http://localhost:7860) with:

```sh
python app.py
```

### Docker

Clone the repo with:

```sh
git clone https://github.com/nbieck/HoldMyNeRF.git 
```

Use `docker-compose` to build and run the containers. Be warned that this will take approx. 45min.

```sh
docker-compose up -d
```

Alternatively without `docker-compose`, build the image:

```sh
docker build -t hold-my-nerf .
```

and run the container:

```sh
docker run --rm --gpus all -p 7860:7860 --name hold-my-nerf -v model:/app/model -v u2net:/root/.u2net hold-my-nerf
```

Access the app on [localhost:7860](http://localhost:7860).

Check [Troubleshooting](#troubleshooting) section for solutions to `insufficient CUDA memory` or `nvcc fatal : Unsupported gpu architecture 'compute_'` and similar.

## Hold your own NeRF

This section explains how to create your own NeRF. In other words, here you'll find instructions on recording a dataset (video), launching the app, using the UI, and obtaining the results.

### Recording a dataset

First off, we'll need a video of the object we want to NeRF. For better results:

1. Film with an empty contrasting background.
2. Avoid covering the object with your hands.
3. Keep the object within frame at all times.
4. Make sure the object fills most of the frame.
5. Make sure the video covers most of the object (360° around it).
6. Don't rotate the object too fast. Rotating it slowly and covering 360° in under 30s is recommended.
7. Avoid transparent and reflective objects.

The tips from [COLMAP Tutorial](https://colmap.github.io/tutorial.html) will help at the camera pose estimation step:

1. Capture images with **good texture**. Avoid completely texture-less images (e.g., a white wall or empty desk). If the scene does not contain enough texture itself, you could place additional background objects, such as posters, etc.
    - Note that we don't care about reconstructing the background since we segment the object, so this applies to the object itself.
3. Capture images at similar **illumination conditions**. Avoid high dynamic range scenes (e.g., pictures against the sun with shadows or pictures through doors/windows). Avoid specularities on shiny surfaces.
4. Capture images with **high visual overlap**. Make sure that each object is seen in at least 3 images – the more images the better.
5. Capture images from **different viewpoints**. Do not take images from the same location by only rotating the camera, e.g., make a few steps after each shot. At the same time, try to have enough images from a relatively similar viewpoint. Note that more images is not necessarily better and might lead to a slow reconstruction process. If you use a video as input, consider down-sampling the frame rate.

Check [Examples](#examples) for examples of different videos and their results.

### Launching the app

If you installed locally, you can start the app with 

```sh
python app.py
```

If you're using Docker, just run the container:

```sh
docker-compose up -d
```

or

```sh
docker run --rm --gpus all -p 7860:7860 --name hold-my-nerf -v model:/app/model -v u2net:/root/.u2net hold-my-nerf
```

The app can be accessed through your browser on `localhost:7860`.

### Using the UI

> Tested on a system with NVIDIA GeForce RTX 2060 and AMD Ryzen 7 4800H.

The UI was designed to be intuitive while also giving you a lot of control over the parameters and results. To start, either **upload or drag-and-drop the video you recorded** in section [Recording a dataset](#recording-a-dataset) into the `Video` field. You can also play the video after uploading it.

Next, you'll want to **fill the `Object Label` text box** with a one-word description of the object you recorded. This can be, for example, 'cube', 'bottle', 'flower', etc. **Click on `Preview Segmentation`** to preview what the segmentation does according to your inputs. The preview is shown in the right side, in the `Segmentation` field under the `Preview` tab. If your object didn't get well segmented, try a different label, or record the video again following the tips in [Recording a dataset](#recording-a-dataset).


![UI preview segmentation](/docs/imgs/ui-preview-segmentation.png)

*Advanced settings: Run Parameters.* These settings define the parameters used by `instant-ngp` when generating the NeRF. In other words, this will influence the generation of the 3D representation of your object.
- Per Image Latents: associates a small embedding factor to each image that's used as an additional input to the network. In other words, makes the model more robust to changes in lighting.
- #Steps: amount of steps to train for. The higher, the longer it'll take to train. Recommended to use `1000` at first and retrain later if necessary.
- Use rembg: Remove background before segmentation. This can either improve or worsen results, depending on the dataset.
- Show Masked Frames: shows the masked frames that were extracted from the video and processed. Useful for checking if all frames were segmented correctly.

If everything is set up, **click on `Submit` to start the pipeline**. Be sure to **click on the `Results` tab** on the right to see the progress. For 720p videos and the default parameters, the pipeline takes approx. 10min to finish.

Once the pipeline is done processing, you'll be able to download the generated 3D mesh and the `instant-ngp` checkpoint under the `Instant-NGP output`. An orbit video is also available for viewing and download under `Orbit Video`.

![UI done](/docs/imgs/ui-results-done.png)

*Advanced settings: Marching cubes resolution.* Defines the resolution of the generated 3D mesh. The higher the resolution, the higher the quality and size of the resulting mesh. Too high values might crash the app. Click on `Regenerate Model` to regenerate the mesh.

*Advanced settings: Video Parameters.* Define different parameters for rendering the 3D NeRF orbit video.
- Width and Height: the width and height of the video.
- FPS: frames per second. 
- Video Length: the duration of the video, in seconds.
- Samples per Pixel: the higher this values is, the higher quality will the rendering be.
- **set the `Video Parameters`** to your preferences and then **click on `Render Video`**. 
- *Warning:* be aware that the higher these values, the longer it'll take to render the video. Default settings render the video in under a minute. Too high values might crash the app.

## Examples

There is also the option of using any of the examples available. Click on one of them to load the example video and its respective label.

The following examples are provided:
- Rubik's cube (handheld): a Rubik's cube being rotated while handheld. White background. Results in partial reconstruction due to COLMAP failing to match all the frames. This occurs because in many of the frames, the cube has significant holes caused by the segmentation. They are also noticeable in the NeRF.
- Flower (handheld): a small flower rotated while handheld. White background. Results in a good reconstruction 90% of the time.

![Cube video](/docs/imgs/cube_video.gif) ![Flower video](/docs/imgs/flower_video.gif) 
*Input video*

![Cube render](/docs/imgs/cube_render.gif)  ![Flower render](/docs/imgs/flower_render.gif)
*Rendered NeRF orbit video*

![Cube mesh](/docs/imgs/cube_mesh.png) ![Flower mesh](/docs/imgs/flower_mesh.png)
*3D Mesh*

## Future work

- [ ] Implement [blind image inpainting](https://arxiv.org/abs/2003.06816#:~:text=Blind%20inpainting%20is%20a%20task,missing%20areas%20in%20an%20image.) into the pipeline.
- [ ] Get funding for Hugging Face deployment.
- [ ] Add toggle buttons for switching between segmentation strategies: 
  - [x] rembg + SEEM;
  - [x] SEEM only;
  - [ ] segmenting background and hands instead of the object.

## Notes

**Interesting discoveries during the development of this project that were implemented into the pipeline:**

- We deceive our camera pose estimator (COLMAP) by creating the illusion of camera movement around an object, while the actual movement is performed by the object itself.​ This is achieved by removing both the background and the hands from the images (or, in other words, by segmenting the object).
- Setting `"aabb_scale": 1` in the `transforms.json` file of the dataset improves the results significantly, as it prevents the NeRF model from extending rays to a much larger bounding box than the unit cube. As the handheld object will always be within the unit cube, this is the ideal value for this parameter and prevents outside floaters and noise.

**Notes on `docker run` arguments:**
- `rm`: doesn't keep the container around. Allows you to simply execute `docker run` again after shutting off the container.
- `--gpus all`: enables GPU use inside the container. Required for CUDA.
- `-p 7860:7860`: connects the host's port 7860 with the container's. Allows access to the Gradio app.
- `--name hold-my-nerf`: gives the container a cool name.
- `-v model:/app/model`: mount the volume at `/app/model`, persisting model checkpoints through container lifecycles. In other words, prevents the container from downloading the checkpoint every time it's booted up. Same for `-v u2net:/root/.u2net/`.

## Troubleshooting

### Insufficient CUDA memory and/or unsupported GPU architecture when building Docker

If you have problems concerning the CUDA architecture or insufficient memory, change the `ARG` variables inside `Dockerfile`.
- `CUDA_ARCHITECTURES=75` supports all GPUs with compute capability of `7.5` or higher (`7.5` is the value for Geforce RTX 2060). If you have an older GPU, you should change this value to your GPU specific value. Check "CUDA-Enabled GeForce and TITAN Products" at https://developer.nvidia.com/cuda-gpus. Settings this to `native` causes the `nvcc fatal : Unsupported gpu architecture 'compute_'` error.
- `NUM_JOBS=4` creates 4 jobs to build `instant-ngp` in parallel. Higher values demand more memory. If you get insufficient memory errors, lower this value.


## Acknowledgements

Thanks to our Professor Philippe Colantoni for providing guidance and scripts for using SEEM, and to our colleagues for the moral support.
