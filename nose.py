import dlib
import cv2
import numpy as np
from math import hypot

#Live Video
cap = cv2.VideoCapture(0)
# The nose image to be masked
nose_img = cv2.imread("noseImg.png")
_, frame = cap.read()

#Getting rows and colums of the picture
rows, cols, _ = frame.shape
# Setting default value for the mask of nose
nose_mask = nose_mask = np.zeros((rows, cols), np.uint8)

# Face detection by default by DLIB
face_detector = dlib.get_frontal_face_detector()
# Face landmarks file for dectecting the nodal points of the face : eg - Nose (top : 29),(left: 31),(right:35)
predictor = dlib.shape_predictor("landmarks.dat")

while True:
    _,frame = cap.read()
    nose_mask.fill(0)
    # Getting gray frame as it reduces the computation time 
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


    faces = face_detector(frame)
    for face in faces:
        landmarks =predictor(gray_frame,face)

        # Getting exact (X,Y) points for nose
        top_nose = (landmarks.part(29).x,landmarks.part(29).y)
        center_nose = (landmarks.part(30).x,landmarks.part(30).y)
        left_nose = (landmarks.part(31).x,landmarks.part(31).y)
        right_nose = (landmarks.part(35).x,landmarks.part(35).y)

        # Calculating the nose widht height by simple mathematical technique 
        nose_width = int(hypot(left_nose[0]-right_nose[0],
                              left_nose[1]-right_nose[1])*1.7)
        nose_height = int(nose_width* 0.74)

        top_left = (int(center_nose[0] - nose_width/2),
                    int( center_nose[1] - nose_height/2))
        bottom_right = (int(center_nose[0] + nose_width/2),
                        int(center_nose[1] + nose_height/2))
        
        # resizing the dog nose as per the size of the user shaope of the user's nose
        dog_nose = cv2.resize(nose_img,(nose_width,nose_height))
        # Asusual conveting it into Gray Frame for less computation
        dog_nose_gray = cv2.cvtColor(dog_nose,cv2.COLOR_BGR2GRAY)

        # creating nose mask of the nose image by implying simple Binary Thresh : https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
        _,nose_mask = cv2.threshold(dog_nose_gray,25,255,cv2.THRESH_BINARY_INV)

        # Creating a rectangle over the nose of the user to be replaced by the nose_img
        nose_area = frame[top_left[1]:top_left[1]+nose_height,
                    top_left[0]:top_left[0]+nose_width]

        # Masking of the nose area 
        nose_area_no_nose = cv2.bitwise_and(nose_area,nose_area,mask=nose_mask)
        # Final nose masked 
        final_nose = cv2.add(nose_area_no_nose,dog_nose)

        # AS live video is generated, final nose is being replaced as per user's live feed 
        frame[top_left[1]:top_left[1]+nose_height,top_left[0]:top_left[0]+nose_width] = final_nose



        # cv2.imshow("nose area",nose_area)
        # cv2.imshow("Nose Dog",dog_nose)
        cv2.imshow("Final Nose",nose_mask)

        # cv2.circle(frame,top_nose,3,(255,0,0),-1)
        

    cv2.imshow("Frame",frame)
   
    # ESC to end the script 
    key = cv2.waitKey(1)
    if key == 27:
        break