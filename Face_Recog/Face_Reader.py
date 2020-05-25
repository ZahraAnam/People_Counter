# import libraries
import sys
import cv2
import time
import face_recognition
from imutils.video import FPS
import imutils
import numpy as np
from threading import Thread


if sys.version_info >= (3, 0):
	from queue import Queue
else:
    from Queue import Queue
"""
class FileVideoStream:
	def __init__(self, path, queueSize=128):
		# initialize the file video stream along with the boolean
		# used to indicate if the thread should be stopped or not
		self.stream = cv2.VideoCapture(path)
		self.stopped = False
		# initialize the queue used to store frames read from
		# the video file
		self.Q = Queue(maxsize=queueSize)
	def start(self):
		# start a thread to read frames from the file video stream
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		# keep looping infinitely
		while True:
			# if the thread indicator variable is set, stop the
			# thread
			if self.stopped:
				return
			# otherwise, ensure the queue has room in it
			if not self.Q.full():
				# read the next frame from the file
				(grabbed, frame) = self.stream.read()
				# if the `grabbed` boolean is `False`, then we have
				# reached the end of the video file
				if not grabbed:
					self.stop()
					return
				# add the frame to the queue
				self.Q.put(frame)
    
	def read(self):
		# return next frame in the queue
		return self.Q.get()

	def more(self):
		# return True if there are still frames in the queue
		return self.Q.qsize() > 0
	
	def stop(self):
		# indicate that thread should be stopped
		self.stopped = True


	
path = r'F:\Face_Recog\videos\video1.mp4'
fvs = FileVideoStream(path).start()
face_locations = []
time.sleep(1.0)
fps = FPS().start()
while fvs.more():
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale (while still retaining 3
	# channels)
	frame = fvs.read()
	frame = imutils.resize(frame, width=450)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	#frame = np.dstack([frame, frame, frame])
	face_locations = face_recognition.face_locations(frame)
	print(face_locations)
	for top, right, bottom, left in face_locations:
        # Draw a box around the face
		cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)
    

	# display the size of the queue on the frame
	cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
		(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	

	# show the frame and update the FPS counter
	cv2.imshow("Frame", frame)
	cv2.waitKey(1)
	fps.update()
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()
"""

def detect_faces(cascade, test_image, scaleFactor = 1.1):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()
    
    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    
    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=1)
    print('Faces found: ', len(faces_rect))
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 5)
        
    return image_copy
# Get a reference to webcam 
haar_cascade_face = cv2.CascadeClassifier(r'C:\Users\zahra\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(r'F:\Face_Recog\videos\video3.mp4')
fps = FPS().start()
# Initialize variables
face_locations = []

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    print(frame.shape)
    if not ret:
        break
	

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    frame = imutils.resize(frame,width=450)
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
	#rgb_frame = np.dstack([rgb_frame,rgb_frame,rgb_frame])

	# Find all the faces in the current frame of video
    #face_locations = face_recognition.face_locations(rgb_frame)

	# Display the results
    #for top, right, bottom, left in face_locations:
	# Draw a box around the face
    #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    faces_rects = detect_faces(haar_cascade_face,rgb_frame)
    
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
