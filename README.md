## background_subtraction.py 
Command line app that can remove the background from a video or folder of images using the rembg library. Call with -h flag for information.

### Requirements 
- (rembg)[https://github.com/danielgatis/rembg]
- numpy == 1.23.5
- PIL
```
pip install rembg numpy==1.23.5
```
**CAUTION**
numpy 1.24 will not work


## Instructions

Remove background from images with:

```sh
python bg_remove.py --img src dst
```

where `src` is the source folder path containing all images, and `dst` is the destination folder for the images without background. The `--img` flag can be changed to `video` if `src` points to a video file.

## SAM - Segment Anything

Requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Follow instructions [here](https://github.com/facebookresearch/segment-anything).

This command install PyTorch with CUDA:

```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

This installs SAM:

```sh
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Extra dependencies:

```sh
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

To run the notebooks:

```sh
pip install jupyter
```