import argparse
import pathlib
import cv2

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()


img_dir = 'output/'
input_list = list(pathlib.Path(img_dir).glob('**/*.jpg'))

for imagePath in input_list:
	# load the image, convert it to grayscale, and compute the
	# focus measure of the image using the Variance of Laplacian
	# method
    image = cv2.imread(str(imagePath))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
    cell_size = [gray.shape[0]//10, gray.shape[1]//10]

    blurry_threshold = 50

    # used for looking up at the region where the laplation is used
    obj_width = [gray.shape[0], 0]
    obj_height = [gray.shape[1], 0]

    # break the image in to cell size to see if the cell contains information
    for w in range(cell_size[0], gray.shape[0], cell_size[0]):
        for h in range(cell_size[1], gray.shape[1], cell_size[1]):
            cell = gray[w-cell_size[0]:w, h-cell_size[1]:h]
            val = cell.mean()
            
            # if the cell is not black
            if val > 0:
                if w-cell_size[0] < obj_width[0]:
                     obj_width[0] = w-cell_size[0]
                elif w > obj_width[1]:
                     obj_width[1] = w

                if h-cell_size[1] < obj_height[0]:
                     obj_height[0] = h-cell_size[1]
                elif h > obj_height[1]:
                     obj_height[1] = h


    fm = variance_of_laplacian(gray[obj_width[0]:obj_width[1], obj_height[0]:obj_height[1]])
    text = "Not Blurry"
    # if the focus measure is less than the supplied threshold,
    # then the image should be considered "blurry"
    if fm < blurry_threshold:
        text = "Blurry"
    # show the image
    cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.imshow("Image", image)
    cv2.imshow("image in focus", image[obj_width[0]:obj_width[1], obj_height[0]:obj_height[1]])
    # press any key to go next image
    key = cv2.waitKey(0)