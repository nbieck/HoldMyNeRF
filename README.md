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
# Windows
pip install -r requirements/windows/torch.txt
pip install -r requirements/windows/r.txt

# Linux
pip install -r requirements/linux/requirements.txt
pip install -r requirements/linux/requirements_git.txt
```

Now, you'll need to build `instant-ngp`. Follow the [instructions in the instant-ngp repository](https://github.com/NVlabs/instant-ngp#building-instant-ngp-windows--linux).

**IMPORTANT** when building `instant-ngp`, ensure that you have the same python environment active as where the requirements were installed.

Finally, start the app on `localhost:7860` with:

```sh
python app.py
```

### Docker

Clone the repo with:

```sh
git clone https://github.com/nbieck/HoldMyNeRF.git 
```

Build the docker image (don't forget the dot `.`). Be warned that this will take approx. 15min.

```sh
docker build -t hold-my-nerf .
```

Run the container:

```sh
docker run --rm --gpus all -p 7860:7860 --name hold-my-nerf hold-my-nerf
```

Access the app on `localhost:7860`.

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

If you installed locally, you can start the app with 

```sh
python app.py
```

If you're using Docker, just run the container:

```sh
docker run -d --gpus all -p 7860:7860 --name hold-my-nerf hold-my-nerf
```

The app can be access through your browser on `localhost:7860`.

### Using the UI

> Tested on a system with NVIDIA GeForce RTX 2060 and AMD Ryzen 7 4800H.

The UI was designed to be intuitive while also giving you a lot of control over the parameters and results. To start, either **upload or drag-and-drop the video you recorded** in section [Recording a dataset](#recording-a-dataset) into the `Video` field. You can also play the video after uploading it.

Next, you'll want to **fill the `Object Label` text box** with a one-word description of the object you recorded. This can be, for example, 'cube', 'bottle', 'flower', etc. **Click on `Preview Segmentation`** to preview what the segmentation does according to your inputs. The preview is shown in the right side, in the `Segmentation` field under the `Preview` tab. If your object didn't get well segmented, try a different label, or record the video again following the tips in [Recording a dataset](#recording-a-dataset).

*Advanced settings: NeRF Parameters.* These settings define the parameters used by `instant-ngp` when generating the NeRF. In other words, this will influence the generation of the 3D representation of your object.
- Steps: amount of steps to train for. The higher, the longer it'll take to train. Recommended to use `1000` at first and retrain later if necessary.

If everything is set up, **click on `Submit` to start the pipeline**. Be sure to **click on the `Results` tab** on the right to see the progress. For 720p videos and the default parameters, the pipeline takes approx. 10min to finish.

Once the pipeline is done processing, you'll be able to see and download the generated 3D mesh and the `instant-ngp` checkpoint. The download buttons are in the top-right of the output fields (they appear once you hover over them).

*Advanced settings: Marching cubes resolution.* Defines the resolution of the generated 3D mesh. The higher the resolution, the higher the quality and size of the resulting mesh. Too high values might crash the app. Click on `Regenerate Model` to regenerate the mesh.

For a downloadable orbit video of the NeRF, **set the `Video Parameters`** to your preferences and then **click on `Render Video`**.
- Width and Height: the width and height of the video.
- FPS: frames per second. 
- Video Length: the duration of the video, in seconds.
- Samples per Pixel: the higher this values is, the higher quality will the rendering be.
  
*Warning:* be aware that the higher these values, the longer it'll take to render the video. Default settings render the video in under a minute. Too high values might crash the app.

## Examples

There is also the option of using any of the examples available. Click on one of them to load the example video and its respective label.

The following examples are provided:
- Rubik's cube (handheld): a Rubik's cube being rotated while handheld. White background. Results in partial reconstruction due to COLMAP failing to match all the frames. This occurs because in many of the frames, the cube has significant holes caused by the segmentation. They are also noticeable in the NeRF.



## Future work

- [ ] Implement [blind image inpainting](https://arxiv.org/abs/2003.06816#:~:text=Blind%20inpainting%20is%20a%20task,missing%20areas%20in%20an%20image.) into the pipeline.

## Notes

Interesting discoveries during the development of this project that were implemented into the pipeline:

- We deceive our camera pose estimator (COLMAP) by creating the illusion of camera movement around an object, while the actual movement is performed by the object itself.​ This is achieved by removing both the background and the hands from the images (or, in other words, by segmenting the object).
- Setting `"aabb_scale": 1` in the `transforms.json` file of the dataset improves the results significantly, as it prevents the NeRF model from extending rays to a much larger bounding box than the unit cube. As the handheld object will always be within the unit cube, this is the ideal value for this parameter and prevents outside floaters and noise.

## Acknowledgements

Thanks to our Professor Philippe Colantoni for providing guidance and scripts for using SEEM, and to our colleagues for the moral support.
