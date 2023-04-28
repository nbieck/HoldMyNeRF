import argparse
import os.path

args = {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove the background from a folder of images or a video.")

    parser.add_argument("input", help="The path to the input video file / image folder")
    parser.add_argument("output", help="The output directory")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", action="store_true", help="process a video")
    input_group.add_argument("--img", action="store_true", help="Process a folder of images")

    args = parser.parse_args()

    if args.video:
        if not os.path.isfile(args.input):
            print("Given input is not an existing file")
            exit()
    elif args.img:
        if not os.path.isdir(args.input):
            print("Given input does not specify an existing directory")
            exit()

from rembg import remove
import cv2

def video_reader(filename):
    cap = cv2.VideoCapture(filename)
    framecount = 0
    ret, frame = cap.read()
    while ret == True:
        yield frame, str(framecount).zfill(4) + ".jpg"
        ret, frame = cap.read()
        framecount += 1

    cap.release()

def image_reader(folder):
    with os.scandir(folder) as it:
        for entry in it:
            if entry.is_file():
                yield cv2.imread(entry.path), entry.name

def process_images(reader, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for img, filename in reader:
        output = remove(img)

        cv2.imwrite(os.path.join(output_dir, filename), output)

if __name__ == "__main__":
    reader = []
    if args.video:
        reader = video_reader(args.input)
    elif args.img:
        reader = image_reader(args.input)

    print(f"Removing background from {args.input} and writing to {args.output}")

    process_images(reader, args.output)