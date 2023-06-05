# :beers: Hold My NeRF!


[Niklas Bieck](https://github.com/nbieck) | [Fabiano Junior Maia Manschein](https://github.com/Fabulani) | [Yuya Takagi](https://github.com/shiohiyoko)

__Jean Monnet University (UJM)__

__Imaging and Light in Extended Reality (IMLEX)__

__[Code](https://github.com/nbieck/HoldMyNeRF)&nbsp;| [Video]()&nbsp;| [Presentation]()__


## Requirements

- [instant-ngp requirements](https://github.com/NVlabs/instant-ngp#requirements)
- [FFmpeg](https://www.ffmpeg.org/)

## Installation

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
pip install -r requirements.txt
```

Now, you'll need to build `instant-ngp`. Follow the [instructions in the instant-ngp repository](https://github.com/NVlabs/instant-ngp#building-instant-ngp-windows--linux).

**IMPORTANT** when building `instant-ngp`, ensure that you have the same python environment active as where the requirements were installed.

## Hold your own NeRF

First off, we'll need a video of the object we want to NeRF. For better results:

1. Film with an empty contrasting background.
2. Avoid covering the object with your hands.
3. Keep the object within frame at all times.
4. Make sure the object fills most of the frame.
5. Make sure the video covers most of the object (360° around it).
6. Don't rotate the object too fast. Rotating it slowly and covering 360° in under 30s is recommended.
7. Avoid transparent and reflective objects.

Here's a quick example:

![Good contrast dataset](/docs/imgs/good_contrast.gif)

Examples of bad videos:

![Bad contrast dataset](/docs/imgs/bad_contrast.gif)

>GIFS GO HERE (side by side)





## Examples

## Future work

- [ ] Implement [blind image inpainting](https://arxiv.org/abs/2003.06816#:~:text=Blind%20inpainting%20is%20a%20task,missing%20areas%20in%20an%20image.) into the pipeline.

## Notes

Interesting discoveries during the development of this project that were implemented into the pipeline:

- We deceive our camera pose estimator (COLMAP) by creating the illusion of camera movement around an object, while the actual movement is performed by the object itself.​ This is achieved by removing both the background and the hands from the images (or, in other words, by segmenting the object).
- Setting `"aabb_scale": 1` in the `transforms.json` file of the dataset improves the results significantly, as it prevents the NeRF model from extending rays to a much larger bounding box than the unit cube. As the handheld object will always be within the unit cube, this is the ideal value for this parameter and prevents outside floaters and noise.

## Acknowledgements


