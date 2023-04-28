from rembg import remove
from PIL import Image
import numpy as np
import cv2

# 動画書き込み変数設定
cap = cv2.VideoCapture('soap.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# writer = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
cnt = 0
ret, frame = cap.read()
while ret == True:
    try:
        output = remove(frame)

        # cv2.imshow('frame',output)
        # writer.write(output)
        cv2.imwrite('output/'+str(cnt).zfill(4)+'.jpg', output)
        cnt += 1
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        ret, frame = cap.read()
    except:
        print("image is none")
cap.release()
# writer.release()
cv2.destroyAllWindows()