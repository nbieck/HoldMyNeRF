import cv2
import os

def video_reader(filename):
    cap = cv2.VideoCapture(filename)
    framecount = 0
    ret, frame = cap.read()
    while ret == True:
        yield frame, str(framecount).zfill(4)
        ret, frame = cap.read()
        framecount += 1

    cap.release()

def image_reader(folder):
    with os.scandir(folder) as it:
        for entry in it:
            if entry.is_file():
                base_name, _ = os.path.splitext(entry.name)
                yield cv2.imread(entry.path), base_name
