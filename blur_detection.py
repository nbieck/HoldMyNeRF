import argparse
import pathlib
import cv2
import os
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt


def varLaplacian(image, blurry_threshold):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    text = "Not Blurry"
    # if the focus measure is less than the supplied threshold,
    # then the image should be considered "blurry"
    if fm < blurry_threshold:
        text = "Blurry"
    # show the image
    cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.imshow("Image", image)

def FFTconv(image):

    results = []
    fft_img = fft.fft2(image)
    fft_img = fft.fftshift(fft_img)

    filter, output = highPassFilter(fft_img, 0)
    spectrum = 20 * np.log(np.abs(output))

    ifft_img = invFFTconv(output)

    results.append(np.abs(filter))
    results.append(np.abs(spectrum))
    results.append(np.abs(output))
    results.append(ifft_img)

    return results


def invFFTconv(image):
    ifft_img = fft.ifftshift(image)
    ifft_img = fft.ifft2(ifft_img)

    return ifft_img.real


def highPassFilter(image, threshold):
    radius = threshold
    center = (image.shape[1]//2, image.shape[0]//2)
    filter = np.ones(image.shape, dtype='uint8')

    filter = cv2.circle(filter, center, radius, 0, -1)

    output = np.multiply(filter, image)
    return filter, output


# def showfft(real, imag):
#     fimg = np.zeros(real.shape, np.complex128)
#     fimg.real = real
#     fimg.imag = imag
#     img = np.abs(fimg)
#     img = img/10
#     cv2.imshow("fft img",img)
#     return img

def showResult(images):
    fig, axes = plt.subplots(1,4, figsize=(30,5))
    for i, img in enumerate(images):
        axes[i].imshow(img)
    
    plt.show()



def BlurDetect(path):
    with os.scandir(path) as it:
        for img in it:
            if img.is_file():
                image = cv2.imread(img.path)
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


                # varLaplacian(gray[obj_width[0]:obj_width[1], obj_height[0]:obj_height[1]], blurry_threshold)
                
                images = FFTconv(gray[obj_width[0]:obj_width[1], obj_height[0]:obj_height[1]])
                showResult(images)
                # press any key to go next image. press q for exit
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    return 


     
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="detect the blur image and delete from the folder")

    parser.add_argument("input", default='./output/', help='The path to the image directory')
    # parser.add_argument("output", help='the output directory')

    args = parser.parse_args()

    BlurDetect(args.input)