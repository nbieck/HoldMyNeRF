from hand_remove import remove
import argparse
import os.path
from imageGenerator import image_reader

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Remove the background from a folder of images or a video.")

#     parser.add_argument("input", help="The path to the input video file / image folder")
#     parser.add_argument("output", help="The output directory")

#     input_group = parser.add_mutually_exclusive_group(required=True)
#     input_group.add_argument("--video", action="store_true", help="process a video")
#     input_group.add_argument("--img", action="store_true", help="Process a folder of images")

#     args = parser.parse_args()

#     if args.video:
#         if not os.path.isfile(args.input):
#             print("Given input is not an existing file")
#             exit()
#     elif args.img:
#         if not os.path.isdir(args.input):
#             print("Given inpu9t does not specify an existing directory")
#             exit()

if __name__ =='__main__':
    input_path = 'test_image/'
    output_path = 'output/contour/'

    reader = image_reader(input_path)
    remove(reader, output_path)
    # FloodFill
