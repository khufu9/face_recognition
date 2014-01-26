import os
import cv
import random

capdevice =0 # 0 for integrated cam 1 for usb-cam

		
def write(font, img, point,string,rgb=(255,255,255)):
	cv.PutText( img, string, point, font, cv.RGB(rgb[0],rgb[1],rgb[2]))

window_name = str(random.randint(100000,200000))	
font =  cv.InitFont( cv.CV_FONT_HERSHEY_SIMPLEX, 0.4, 0.4, 0.0, 1, cv.CV_AA )
	
capture = cv.CreateCameraCapture(0)
img = cv.QueryFrame(capture)
cv.NamedWindow(window_name,1)
cv.ResizeWindow(window_name, img.width, img.height)
	
face_cascade = cv.Load("../haarcascade_frontalface_alt.xml")
haar_scale = 1.2
min_neighbors = 2
haar_flags = 0
min_target_size = (20,20)

port = cv.CreateImage( (134,179),8,3 )
offset = 0

while True:
	img = cv.QueryFrame( capture )			

	cr = 0.50	
	icr = 1.0/cr

	gray = cv.CreateImage( (img.width, img.height), 8, 1 )
	small = cv.CreateImage( (cv.Round(img.width*cr), cv.Round(img.height*cr)), 8, 1)

	# covert image to gray
	cv.CvtColor( img, gray, cv.CV_BGR2GRAY )

	# scale input image
	cv.Resize( gray, small, cv.CV_INTER_LINEAR )

	# equalize histogram
	cv.EqualizeHist( small, small)

	suspects = cv.HaarDetectObjects( small, face_cascade, cv.CreateMemStorage(0), haar_scale, min_neighbors, haar_flags, min_target_size )

	if len(suspects) > 0:
		num = 1	
		for ((x,y,w,h), n) in suspects:
			top_left = (int(x*icr)+int(5*icr), int(y*icr)+int(10*icr))
			bottom_right = (int((x+w) * icr)-int(5*icr), int((y+h)*icr)-int(0*icr))
			x = int(x*icr)+15
			y = int(y*icr)+10
			w = int(w*icr)-15
			h = int(h*icr)-10

			cv.SetImageROI( img, (x,y,w,h) )
			face = cv.CreateImage((w,h),8,3)
			gray_face = cv.CreateImage((134,179),8,1)
			cv.Copy(img,face)
			cv.ResetImageROI(img)
			cv.Resize(face,port,cv.CV_INTER_LINEAR)
			cv.CvtColor(port,gray_face,cv.CV_BGR2GRAY)
			cv.EqualizeHist(gray_face,gray_face)
			cv.SetImageROI( img, (10+offset,img.height-189,134,179) )
			cv.Copy(port,img)
			write(font,img,(5,160),"kaka",(255,255,255))
			cv.Rectangle(img, (0,0),(134-2,179-2), cv.RGB(30,255,30),2,8,0)

			focus_face = cv.CloneImage(port)
			cv.ResetImageROI(img)

			#offset += 144
			#num += 1
	
	cv.ShowImage(window_name,img)
	key = cv.WaitKey(10)

	if key == 32: #space
		name = raw_input("Enter name: ")
		cv.SaveImage("./"+name+".jpg", gray_face)
		print "Portrait was saved as: ./"+name+".jpg"
	else:
		print key


	#
	#key = cv.WaitKey(10)
	#
	#if key == 113:
	#	scan.killWindow()
	#	break
	#else:
	#	if key != -1:
	#		print key

