import sys
import cv2
import time
import face_recognition
from imutils.video import FPS
import imutils
import numpy as np
from mtcnn.mtcnn import MTCNN
from threading import Thread


if sys.version_info >= (3, 0):
	from queue import Queue
else:
    from Queue import Queue


haar_cascade_face = cv2.CascadeClassifier(r'C:\Users\zahra\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

def deep_face_count(path):
    video_capture = cv2.VideoCapture(path)
    fps = FPS().start()
    # Initialize variables
    model = MTCNN()
    face_count = []
    loc=0
    
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        if not ret:
            break
        

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        frame = imutils.resize(frame,width=450)
        rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        faces = model.detect_faces(rgb_frame)
        k=0
        for face in faces:
            x,y,w,h = face['box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            k+=1
        if k!=0 and loc!=0:
            if face_count[loc-1]==k:
                continue
            else:
                face_count = np.append(face_count,k)
                loc+=1
        if k!=0 and loc==0:
            face_count = np.append(face_count,k)
            loc+=1
        

        
        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    return face_count