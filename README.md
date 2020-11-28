# face-recognition
import cv2
from random import randrange

# load some pre trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose  an image to detect faces in
# img = cv2.imread('katrinaa.png')

# to capture video from webcam
webcam = cv2.VideoCapture(0)
key = cv2.waitKey(1)

## iterate  forever over frames
while True:

    ####Read the  current frame
    successful_frame_read, frame = webcam.read()

    # Must convert to grey-scale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # DETECT faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # draw rectangle around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256),10)

     cv2.imshow("Faces found", frame)
     cv2.waitKey(1) 
        ### stop if Q key is pressed
        if key == 81 or key == 113:
            break

    ### release the videoCapture object
    webcam.release()

print("code completed")
