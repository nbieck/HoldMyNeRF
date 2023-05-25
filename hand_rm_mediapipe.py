import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import cv2
import os


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
	hand_landmarks_list = detection_result.hand_landmarks
	handedness_list = detection_result.handedness
	annotated_image = np.copy(rgb_image)

	# Loop through the detected hands to visualize.
	for idx in range(len(hand_landmarks_list)):
		hand_landmarks = hand_landmarks_list[idx]
		handedness = handedness_list[idx]

		# Draw the hand landmarks.
		hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
		hand_landmarks_proto.landmark.extend([
			landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
		])
		# print(hand_landmarks_proto)
		solutions.drawing_utils.draw_landmarks(
			annotated_image,
			hand_landmarks_proto,
			solutions.hands.HAND_CONNECTIONS,
			solutions.drawing_styles.get_default_hand_landmarks_style(),
			solutions.drawing_styles.get_default_hand_connections_style())

		# Get the top left corner of the detected hand's bounding box.
		height, width, _ = annotated_image.shape
		x_coordinates = [landmark.x for landmark in hand_landmarks]
		y_coordinates = [landmark.y for landmark in hand_landmarks]
		text_x = int(min(x_coordinates) * width)
		text_y = int(min(y_coordinates) * height) - MARGIN

		# Draw handedness (left or right hand) on the image.
		# cv2.putText(annotated_image, f"{handedness[0].category_name}",
		# 			(text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
		# 			FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
		
		cv2.circle(annotated_image, (int(x_coordinates[0]*width), int(y_coordinates[0]*height)) , 10, 0, 5)
		mask = MaskHand(rgb_image, [int(x_coordinates[0]*width), int(y_coordinates[0]*height)])

		annotated_image[mask==255] = 0
		# cv2.imshow("masked hand", annotated_image)

	return annotated_image

def MaskHand(image, hand_landmarks):

	hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	skin_color = np.mean(np.mean(hsv_img[hand_landmarks[1]-5:hand_landmarks[1]+5, hand_landmarks[0]-5:hand_landmarks[0]+5], axis=1), axis=0)
	# print("raw value", hsv_img[hand_landmarks[1]-5:hand_landmarks[1]+5, hand_landmarks[0]-5:hand_landmarks[0]+5])
	print("skincolor val", skin_color)
	
	lower_thres = np.array([skin_color[0]-2, 50, 50])
	high_thres = np.array([skin_color[0]+3, 200, 200])
	f_img = cv2.inRange(hsv_img, lower_thres, high_thres)
	cv2.imshow("skin filtered img ", f_img)
	
	# image[f_img] = 0
	# cv2.imshow("masked image", image)
	return f_img
	
if __name__ == '__main__':

	root = os.getcwd()
	image_path = root+'/test_image/IMG_20230504_112731.jpg'

	image = cv2.imread(image_path)
	image = cv2.resize(image, (int(image.shape[1]/3), int(image.shape[0]/3)))

	mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

	# STEP 2: Create an HandLandmarker object.
	base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
	options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
	detector = vision.HandLandmarker.create_from_options(options)

	# STEP 4: Detect hand landmarks from the input image.
	detection_result = detector.detect(mp_image)

	# STEP 5: Process the classification result. In this case, visualize it.
	annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
	
	if annotated_image is not None:
		cv2.imshow("image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

	if cv2.waitKey(0) == ord('q'):
		cv2.destroyAllWindows()