import numpy as np
import cv2
import os

def Zero_crossing(image):
    z_c_image = np.zeros(image.shape)
    
    # For each pixel, count the number of positive
    # and negative pixels in the neighborhood
    
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            negative_count = 0
            positive_count = 0
            neighbour = [image[i+1, j-1],image[i+1, j],image[i+1, j+1],image[i, j-1],image[i, j+1],image[i-1, j-1],image[i-1, j],image[i-1, j+1]]
            d = max(neighbour)
            e = min(neighbour)
            for h in neighbour:
                if h>0:
                    positive_count += 1
                elif h<0:
                    negative_count += 1


            # If both negative and positive values exist in 
            # the pixel neighborhood, then that pixel is a 
            # potential zero crossing
            
            z_c = ((negative_count > 0) and (positive_count > 0))
            
            # Change the pixel value with the maximum neighborhood
            # difference with the pixel

            if z_c:
                if image[i,j]>0:
                    z_c_image[i, j] = image[i,j] + np.abs(e)
                elif image[i,j]<0:
                    z_c_image[i, j] = np.abs(image[i,j]) + d
                
    # Normalize and change datatype to 'uint8' (optional)
    z_c_norm = z_c_image/z_c_image.max()*255
    z_c_image = np.uint8(z_c_norm)

    return z_c_image

def maxArea(contours):
    ci, max_area = -1, 0
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        
        if area > max_area:
            max_area = area
            ci = i
    return contours[ci]

def remove(reader, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for img, filename in reader:
        img = cv2.resize(img, (int(img.shape[1]/5), int(img.shape[0]/5)))
        
        # img = cv2.bitwise_not(img)
        lower_thresh = np.array([0, 0, 0])
        upper_thresh = np.array([25, 255, 255])

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, lower_thresh, upper_thresh)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 50, 255)
        # ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mcnt = maxArea(contours=contours)

        hull = cv2.convexHull(mcnt)
        cv2.drawContours(img, hull, -1, (0,255,0), 3)
        cv2.drawContours(img, contours, -1, (255,0,0), 1)


        blur = cv2.GaussianBlur(gray, (5,5), 0)
        kernel_lap8 = np.array([[1, 1, 1],
                                [1, -8, 1], 
                                [1, 1, 1]])
        kernel_lap4 = np.array([[0, 1, 0],
                                [1, -4, 1], 
                                [0, 1, 0]])
        lap4 = cv2.filter2D(blur, cv2.CV_64F, kernel_lap4)
        lap_blur = cv2.GaussianBlur(lap4, (3,3), 0)
        # lap8 = cv2.filter2D(blur, cv2.CV_64F, kernel_lap8)
        lap5 = cv2.Laplacian(blur, cv2.CV_64F, ksize=7)
        
        # zero_img = Zero_crossing(lap)
        
        # cv2.imwrite(os.path.join(output_dir, filename)+'.png', output)
        # cv2.imshow("gray image", blur )
        # cv2.imshow("canny", canny)
        # cv2.imshow("thresh image", thresh)
        # cv2.imshow("contour",contours)
        cv2.imshow("image", img)
        cv2.imshow("hsv mask", mask_hsv)
        cv2.imshow("laplasian 4", lap4)
        cv2.imshow("lap -> blur", lap_blur)
        # cv2.imshow("laplasian 8", lap5)
        cv2.waitKey(0)


def image_reader(folder):
    with os.scandir(folder) as it:
        for entry in it:
            if entry.is_file():
                base_name, _ = os.path.splitext(entry.name)
                yield cv2.imread(entry.path), base_name


