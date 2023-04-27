from rembg import remove
from PIL import Image
import numpy as np
import cv2

#動画読み込み33
#背景差分のアルゴリズム？を取得。KNNやMOG2
# fgbg = cv2.createBackgroundSubtractorKNN()
# fgbg = cv2.createBackgroundSubtractorMOG2()

# 動画書き込み変数設定
cap = cv2.VideoCapture('soap.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
ret, frame = cap.read()
while ret == True:
    try:
        output = remove(frame)

        cv2.imshow('frame',output)
        writer.write(output)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        ret, frame = cap.read()
    except:
        print("image is none")
cap.release()
writer.release()
cv2.destroyAllWindows()