''' 
For predicting the sign language digit images using trained model from mobile_net.py file
By first loading the trained model
Using open cv to use device camera 
Preprocessing the images in similar manner as done in training 
Making predictions on real time video
Note : Model was trained on images with white background try the same while testing

'''
import os
import cv2
import numpy as np
from keras.models import load_model
from keras.applications.mobilenet import preprocess_input

def prepare_image(image):
	resized_image = cv2.resize(image, (224,224))
	expanded_image = np.expand_dims(resized_image, axis=0)
	return preprocess_input(expanded_image)

	
def predict_image(camera_captured, model, dig_list):
	prepared_image = prepare_image(camera_captured)
	prediction = model.predict(prepared_image)
	max_prob = 0
	dig = 0
	arr = prediction[0]
	for i in range(10):
		if arr[i] > max_prob:
			max_prob = arr[i]
			dig = i
	return dig_list[dig]

def main():
	print(os.getcwd())
	dig_list = ['Zero','One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
		
	# loading the model
	model = load_model(os.getcwd()+'/sign_lang_digits/models/mobile_net.h5')

	# capturing video 0 to specify system camera
	video = cv2.VideoCapture(0)

	ans = "Taking Input from Camera"
	counter = 0
	while True:
		counter += 1
		check, cam_img = video.read()
		if counter == 40:
			ans = predict_image(cam_img,model,dig_list)
			print(ans)
			counter = 0
		
		output_img = cv2.putText(cam_img, ans, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
		cv2.imshow("Output", output_img)
		
		key = cv2.waitKey(10)

		# using q to close the recording
		if key == ord('q'):
			break

	video.release()
	cv2.destroyAllWindows

if __name__ == '__main__':
	main()