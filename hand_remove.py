import numpy as np
import cv2
import os
import queue
import matplotlib.pyplot as plt

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

def isAvaliable(open_space, coord):
    """
    Search through open_space if the coord is in the list
    input:
        open_space: list of xy coord [[xcoord], [ycoord]]
        coord: coordinate to lookup

    output:
        idx: index for open_space. If its not there, it returns -1
    """
    ref_x = np.where(open_space[0] == coord[0])
    ref_y = np.where(open_space[1] == coord[1])
    idx = -1
    
    for x in ref_x[0]:
        if x in ref_y[0]:
            idx = x

    return idx

def poprow(my_array,pr):
    """ row popping in numpy arrays
    Input: my_array - NumPy array, pr: row index to pop out
    Output: [new_array,popped_row] """
    i = pr
    pop = my_array[i]
    new_array = np.vstack((my_array[:i],my_array[i+1:]))
    return [new_array,pop]

def popcol(my_array,pc):
    """ column popping in numpy arrays
    Input: my_array: NumPy array, pc: column index to pop out
    Output: [new_array,popped_col] """
    i = pc
    pop = my_array[:,i]
    new_array = np.hstack((my_array[:,:i],my_array[:,i+1:]))
    return [new_array,pop]

def FloodFill(mask: cv2.Mat)->np.ndarray:
    """
    FloodFill function
    input: 
        mask : ch1 matrix 
    output:
        uint8 mat: labeled matrix
    """

    result = np.zeros((mask.shape[0], mask.shape[1]), dtype='uint8')
    # node to contain the next refrence pixel
    node = queue.Queue()
    label_num = 1

    # list of xy coordinate of pixel
    open_space = np.array(np.where(mask <= 0))
    open_space, xy = popcol(open_space, 0)
    node.put(xy)
    print(mask.max())
    print(open_space[0].shape[0]==0, node.empty())


    while (open_space[0].shape[0] != 0) or (not node.empty()):
        if node.empty():
            open_space, val = popcol(open_space, 0)
            node.put(val)
            label_num += 1
        else:
            xy = node.get()

            # north
            north_idx = isAvaliable(open_space=open_space, coord=[xy[0], xy[1]-1])
            if north_idx >= 0:
                open_space, coord = popcol(open_space, north_idx)
                node.put(coord)
            # south
            south_idx = isAvaliable(open_space=open_space, coord=[xy[0], xy[1]+1])
            if south_idx >= 0:
                open_space, coord = popcol(open_space, south_idx)
                node.put(coord)
            # west
            west_idx = isAvaliable(open_space=open_space, coord=[xy[0]-1, xy[1]])
            if west_idx >= 0:
                open_space, coord = popcol(open_space, west_idx)
                node.put(coord)
            # east
            east_idx = isAvaliable(open_space=open_space, coord=[xy[0]+1, xy[1]])
            if east_idx >= 0:
                open_space, coord = popcol(open_space, east_idx)
                node.put(coord)

            result[xy[0], xy[1]] = label_num

            # # north
            # east_idx = isAvaliable(open_space=open_space, coord=[xy[0]+1, xy[1]])
            # if north_idx >= 0:
            #     node.put([open_space[0].pop(east_idx), open_space[1].pop(east_idx)])
            

    # # look up north, south, west, east to see if they're labeled
    # for y, x in zip(open_space[1], open_space[0]):
    #     # north
    #     if y-1 >= 0:
    #         if result[x][y-1] != 0:
    #             result[x][y] = result[x][y-1]
    #         else:
    #             result[x][y] = label_num
    #             label_num += 1
    #     # south
    #     elif y+1 < height:
    #         if result[x][y+1] != 0:
    #             result[x][y] = result[x][y+1]
    #         else:
    #             result[x][y] = label_num
    #             label_num += 1
    #     # west
    #     elif x-1 <= 0:
    #         if result[x-1][y] != 0:
    #             result[x][y] = result[x-1][y]
    #         else:
    #             result[x][y] = label_num
    #             label_num += 1
    #     # east
    #     elif x+1 < width:
    #         if result[x+1][y] != 0:
    #             result[x][y] = result[x+1][y]
    #         else:
    #             result[x][y] = label_num
    #             label_num += 1
        

        
    # for y in range(mask.shape[1]):
    #     for x in range(mask.shape[0]):
    #         if mask[y][x] == 1:
    #             open_space[y][x] = 1
    #         else:
    #             mask[y][x] = 
                

    return result 

def remove(reader:list[cv2.Mat, str], output_dir:str):
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

        open_space = FloodFill(lap_blur)
        print("label num", open_space.max())
        cv2.imshow("lap -> blur", open_space)
        
        
        # cv2.imwrite(os.path.join(output_dir, filename)+'.png', output)
        # cv2.imshow("gray image", blur )
        # cv2.imshow("canny", canny)
        # cv2.imshow("thresh image", thresh)
        # cv2.imshow("contour",contours)
        
        # cv2.imshow("image", img)
        cv2.imshow("hsv mask", mask_hsv)
        cv2.imshow("laplasian 4", lap_blur)
        # cv2.imshow("laplasian 8", lap5)

        cv2.waitKey(0)

        # plt.figure()
        # plt.hist(lap_blur, 40)
        # plt.show()

