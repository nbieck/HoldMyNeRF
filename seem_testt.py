import sys
import os
import tempfile
sys.path.append(os.path.join(os.path.dirname(__file__), "dependencies/SEEM/demo_code"))
# os.chdir(os.path.join(os.getcwd(), 'dependencies/SEEM/demo_code'))
from seem_extraction import SEEMPipeline, SEEMPreview
    

import argparse
import os.path
import cv2

args = {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove the background from a folder of images or a video.")

    parser.add_argument("input", help="The path to the input video file / image folder")
    parser.add_argument("output", help="The output directory")

    args = parser.parse_args()


if __name__ == "__main__":

    SEEMPipeline(args.input, args.output, "cube")
    file = SEEMPreview(args.input, args.output, 'cube')
    cv2.imshow(file)