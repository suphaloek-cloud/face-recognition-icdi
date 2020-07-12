# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from datetime import datetime  #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
from ftplib import FTP 		#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
import time, threading
import requests
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

def uploadfile_ftp():							#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	now = datetime.now()
	times = now.strftime("%d%m%Y_%H%M%S")

	ftp = FTP('192.168.2.227')
	ftp.login('face', 'icdi2020')
	ftp.cwd('storage_images') # change into directory
	name_dir = name_ftp
	global file_name
	file_name = times+'.png'

	if name_dir in ftp.nlst() : #check if
		print ('YES')
		ftp.cwd(name_dir+"/checkIN")  # change into directory
		with open('IMG_OUT/OUTPUT.png','rb') as file:           # file to send
			ftp.storbinary(f"STOR {file_name}", file)
		ftp.retrlines('LIST') #list directory contents

	else : 
		print ('NO')
		ftp.mkd(name_dir) #Create a new directory called foo on the server.
		ftp.mkd(name_dir+'/checkIN')
		ftp.cwd(name_dir+"/checkIN")  # change into directory
		with open('IMG_OUT/OUTPUT.png','rb') as file:           # file to send
			ftp.storbinary(f"STOR {file_name}", file)
		ftp.retrlines('LIST') #list subdirectory contents

	ftp.quit()
	ftp.close() #close connection

def lineNotify(message):
    payload = {'message':message}
    return _lineNotify(payload)

def notifyFile(name,filename):
    file = {'imageFile':open(filename,'rb')}
    payload = {'message': name+' Check in from ICDI'}
    return _lineNotify(payload,file)

def notifyPicture(url):
    payload = {'message':" ",'imageThumbnail':url,'imageFullsize':url}
    return _lineNotify(payload)

def notifySticker(stickerID,stickerPackageID):
    payload = {'message':" ",'stickerPackageId':stickerPackageID,'stickerId':stickerID}
    return _lineNotify(payload)

def _lineNotify(payload,file=None):
    import requests
    url = 'https://notify-api.line.me/api/notify'
    #token = 'iOlY60l0cojfvypypS6xC0ln3X0b5TtFKz1jUTryDl0'	#EDIT
    token = '8Mk2ufXnQDV97D3wFSLGHTGVGvtSz3Pu7MiqG8jOw9V'
    headers = {'Authorization':'Bearer '+token}
    return requests.post(url, headers=headers , data = payload, files=file)

def database_sql():
	#sent to database
	url = 'http://192.168.2.227:3000/api/face/'
	myobj = {"user_id":  Employee_code, "user_name": Employee_name, "user_pic": "/storage_images/"+name_ftp+"/"+file_name }
	x = requests.post(url,json = myobj)

def thread_end_process():							#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	cv2.imshow('scanned',cropped)
	#uploadfile_ftp()                                #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	#database_sql()									#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	cv2.waitKey(1)
	time.sleep(4)
	cv2.destroyWindow('scanned') 
	#sent to LINE
	notifyFile(name,'IMG_OUT/OUTPUT.png')          #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def thread_embedder(face,embedder):
	faceBlob = cv2.dnn.blobFromImage(cv2.resize(face,(96, 96)), 1.0 / 255, (96, 96), (0, 0, 0),swapRB=True, crop=False)
	embedder.setInput(faceBlob)
	global vec
	vec = embedder.forward()
	# perform classification to recognize the face
	global preds
	preds = recognizer.predict_proba(vec)[0]

def thread_read_video(vs):
	global hasFrame
	global frame
	hasFrame,frame = vs.read()


def setup_program():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	# ap.add_argument("-d", "--detector", required=True,
	# 	help="path to OpenCV's deep learning face detector")
	# ap.add_argument("-m", "--embedding-model", required=True,
	# 	help="path to OpenCV's deep learning face embedding model")
	# ap.add_argument("-r", "--recognizer", required=True,
	# 	help="path to model trained to recognize faces")
	# ap.add_argument("-l", "--le", required=True,
	# 	help="path to label encoder")
	ap.add_argument("-c", "--confidence", type=float, default=0.6,
		help="minimum probability to filter weak detections")
	global args
	args = vars(ap.parse_args())

	# load our serialized face detector from disk
	print("[INFO] loading face detector...")
	# protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
	# modelPath = os.path.sep.join([args["detector"],
	# 	"res10_300x300_ssd_iter_140000.caffemodel"])
	#protoPath = os.path.sep.join(["face_detection_model/deploy.prototxt"])
	#modelPath = os.path.sep.join(["face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"])
	faceProto="opencv_face_detector.pbtxt"
	faceModel="opencv_face_detector_uint8.pb"
	global detector
	detector = cv2.dnn.readNet(faceModel,faceProto)
	#detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
	#detector.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

	# load our serialized face embedding model from disk and set the
	# preferable target to MYRIAD
	print("[INFO] loading face recognizer...")
	# embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])
	global embedder
	embedder = cv2.dnn.readNetFromTorch("face_embedding_model/openface_nn4.small2.v1.t7")
	#embedder.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

	# load the actual face recognition model along with the label encoder
	# recognizer = pickle.loads(open(args["recognizer"], "rb").read())
	# le = pickle.loads(open(args["le"], "rb").read())
	global recognizer
	recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
	global le
	le = pickle.loads(open("output/le.pickle", "rb").read())

	# initialize the video stream, then allow the camera sensor to warm up
	print("[INFO] starting video stream...")
	#vs = VideoStream(src=0).start()
	#vs = VideoStream(usePiCamera=True).start()
	global vs
	vs = cv2.VideoCapture(0)


if __name__ == "__main__":
	#setup firt time start program
	setup_program()
	##############################
	time.sleep(1.0)
	(h, w) = (None, None)
	# start the FPS throughput estimator
	fps = FPS().start()
	#list array name
	list_CheckIN = []
	Temp_CheckFace = []
	#Check_FacePerson = 
	# loop over frames from the video file stream
	people = 0
	#start_time[people] = time.time()
	timeCount = 0
	times = {}
	objects = {}
	check = 0
	Sec = 0
	while True:
		status = "Not found"
		detail = "Unknow"
		# grab the frame from the threaded video stream
		hasFrame,frame = vs.read()
		#thread_frame = threading.Thread(target=thread_read_video(vs))
		#thread_frame.start()
		frame = frame[75:400, 100:600]
		# resize the frame to have a width of 600 pixels (while
		# maintaining the aspect ratio), and then grab the image
		# dimensions
		frame = imutils.resize(frame, width=1200)
		if w is None or h is None:
			(h, w) = frame.shape[:2]

		if not hasFrame:
			cv2.waitKey()
			break

		# construct a blob from the image
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(frame, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)

		# apply OpenCV's deep learning-based face detector to localize
		# faces in the input image
		detector.setInput(imageBlob)
		detections = detector.forward()

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the prediction
			confidence = detections[0, 0, i, 2]
			# filter out weak detections
			if confidence > args["confidence"]:
				# compute the (x, y)-coordinates of the bounding box for
				# the face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# extract the face ROI
				face = frame[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

				# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue

				# construct a blob for the face ROI, then pass the blob
				# through our face embedding model to obtain the 128-d
				# quantification of the face
				
				thread_faceBlob = threading.Thread(target=thread_embedder(face,embedder))
				thread_faceBlob.start()
				#thread_faceBlob.join()

				j = np.argmax(preds)
				#global proba
				proba = preds[j]
				#global name
				name = le.classes_[j]
				global name_ftp
				name_ftp = le.classes_[j]                    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,,
				name = name.replace("_"," ")

				if name in list_CheckIN :
					#print('have in list')
					status = "Successful Scan"
					detail = name
					Sec = 0

				# draw the bounding box of the face along with the
				# associated probability

				cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
				#detail = 'unknow'
				if proba > 0.95 and name != 'unknown' and name not in list_CheckIN:
					#cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
					Sec += 1
					print(Sec)
					status = "Scanning"
					detail = name+" ?"
					if(Sec >= 15 ):
						text = "{}: {:.2f}%".format(name, proba * 100)
						print(text)
						y = startY - 10 if startY - 10 > 10 else startY + 10
						#cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
						cv2.putText(frame, text, (startX, y),
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
						personIMG = cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
						cv2.imwrite('IMG_OUT/OUTPUT.png' , personIMG)                     #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
						cropped = frame[startY:endY,startX:endX]
						#cropped = cv2.resize(cropped, (150, 150))
						#cropped = cv2.resize(cropped, (300, 100))
						# if name in list_CheckIN:
						# 	print('have in list')
						# 	detail = "Scaned"
						# 	time.sleep(1)
								
						if name not in list_CheckIN:
							split_txt = name.split(" ")
							Employee_code = split_txt[0]
							Employee_name = split_txt[1]
							#print(Employee_name)

						#sent to LINE	
							#notifyFile(name,'lineIMG/OUTPUT.png')
							#lineNotify(name+' Check in from ICDI')

							list_CheckIN.append(name)
							people = len(list_CheckIN)
							status = "Successful Scan"
							detail = name
							print(people)
							print('add in list')
							Sec = 0
							#cv2.imshow('scanned',cropped)
							thread_end_process = threading.Thread(target=thread_end_process)
							thread_end_process.start()
							#threading.Timer(4, thread_showcrop).start()
							
						print(list_CheckIN)
					
		info = [
			("Identify", detail),
			("Status", status),
		]
		if frame is not None:
		# loop over the info tuples and draw them on our frame
			for (i, (k, v)) in enumerate(info):
				text = "{}: {}".format(k, v)
				cv2.putText(frame, text, (10, h - ((i * 20) + 20)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)	

		# update the FPS counter
		fps.update()
		
		# show the output frame
		#Fullscreen
		# cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
		# cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("r"):
			list_CheckIN = []
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			# do a bit of cleanup
			# stop the timer and display FPS information
			fps.stop()
			print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
			print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
			cv2.destroyAllWindows()
			#vs.stop()
			break