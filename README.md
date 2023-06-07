# :beers: Hold My NeRF!


[Niklas Bieck](https://github.com/nbieck) | [Fabiano Junior Maia Manschein](https://github.com/Fabulani) | [Yuya Takagi](https://github.com/shiohiyoko)

__Jean Monnet University (UJM)__

__Imaging and Light in Extended Reality (IMLEX)__

__[Code](https://github.com/nbieck/HoldMyNeRF)&nbsp;| [Video]()&nbsp;| [Presentation]()__

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
- [Acknowledgements](#acknowledgements)


## Requirements

For a local installation:
- [instant-ngp requirements](https://github.com/NVlabs/instant-ngp#requirements)
- [FFmpeg](https://www.ffmpeg.org/)

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
pip install -r torch.txt
pip install -r r.txt
```

Now, you'll need to build `instant-ngp`. Follow the [instructions in the instant-ngp repository](https://github.com/NVlabs/instant-ngp#building-instant-ngp-windows--linux).

**IMPORTANT** when building `instant-ngp`, ensure that you have the same python environment active as where the requirements were installed.

### Docker

Either clone the repo with

```sh
git clone https://github.com/nbieck/HoldMyNeRF.git 
```

or download the `Dockerfile`.

Build the docker image (don't forget the dot `.`):

```sh
docker build -t hold-my-nerf .
```

Run the container:

```sh
docker run -p 7860:7860 --name hold-my-nerf hold-my-nerf
```

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

Here's a quick example:

![Good contrast dataset](/docs/imgs/good_contrast.gif)

Examples of bad videos:

![Bad contrast dataset](/docs/imgs/bad_contrast.gif)

>GIFS GO HERE (side by side)


### Launching the app

### Using the UI


## Examples

## Future work

- [ ] Implement [blind image inpainting](https://arxiv.org/abs/2003.06816#:~:text=Blind%20inpainting%20is%20a%20task,missing%20areas%20in%20an%20image.) into the pipeline.

## Notes

Interesting discoveries during the development of this project that were implemented into the pipeline:

- We deceive our camera pose estimator (COLMAP) by creating the illusion of camera movement around an object, while the actual movement is performed by the object itself.​ This is achieved by removing both the background and the hands from the images (or, in other words, by segmenting the object).
- Setting `"aabb_scale": 1` in the `transforms.json` file of the dataset improves the results significantly, as it prevents the NeRF model from extending rays to a much larger bounding box than the unit cube. As the handheld object will always be within the unit cube, this is the ideal value for this parameter and prevents outside floaters and noise.

## Acknowledgements

Thanks to our Professor Philippe Colantoni for providing guidance and scripts for using SEEM, and to our colleagues for the moral support.
