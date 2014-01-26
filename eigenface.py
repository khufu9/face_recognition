# =====================================
#
# Author: Indrome Technologies
# Licence: MIT 
#
# To use this library you must do two things:
#
# 1. Train the recognizer by calling train("/home/foo/bar/baz/")
# with a path to the folder containing your target faces.
#
# 2. Perform recognition by calling lookup(np_map) with a 
# numpy-matrix representation of a face
#
# Check list:
# - Face must have the same dimensions as the images used in training.
#
# References:
# http://www.scholarpedia.org/article/Eigenfaces
#
# =====================================

import os, cv, re
import operator
import numpy as np

def vec_from_img_path(img_path):
	img = np.asarray( cv.LoadImageM(img_path , cv.CV_LOAD_IMAGE_GRAYSCALE) )
	return np.reshape( img[:,:], (reduce(operator.mul, img.shape), 1))
	
def compute_eigenface(UT,v,X):
	p = np.subtract(v,X)
	eigf = np.dot(UT,p)
	return eigf


def file_list_from_path(db_path="./faces/"):
	file_suffix_matcher = re.compile(".jpg|.jpeg|.jpe|.pgm|.ppm|.pbm|.png|.tiff|.tif")
	files = map(lambda x: db_path+x, os.listdir(db_path))
	return files

# =====================================
#
# Training
#
# =====================================
def train(file_list):

	print file_list

	# Load files as cvMats
	images = [ np.asarray(cv.LoadImageM(file_path, cv.CV_LOAD_IMAGE_GRAYSCALE)) for file_path in file_list ]
	
	# Reshape and stack images as columns of a huge matrix
	images = np.asarray(map(lambda M: np.reshape(M[:,:],reduce(operator.mul, M.shape), 1), images))
	R = np.matrix(images.T)

	print "Training: computing eigenface basis"
	X = np.mean(R,axis=1)
	A = np.subtract(R,X)
	D,V = np.linalg.eig( A.T*A )

	print "Training: sorting by descending eigen value"
	ind = D.argsort()[::-1]
	[D,V] = map(np.array,zip(*filter(lambda x: x[0]>0.0,zip(np.real(D[ind]),V[:,ind].T))))
	V = V.T

	print "Training: computing eigenfaces"
	D = np.sqrt(1.0/D)
	D = np.diag(D)
	print "A",A.shape,"V",V.shape,"D",D.shape
	U = A*V*D

	print "Training: Eigenbasis axes",U.shape[1]

	O = None
	for v in R.T:
		if O == None:
			O = compute_eigenface(U.T,v.T,X)
		else:
			O = np.hstack((O, compute_eigenface(U.T,v.T,X)))

	return O, U.T, X


# =====================================
# Recognition
#
# X is the is the average face
# U is the eigenface basis.
# 
# =====================================
def recognize(O,U,v,X):

	o = compute_eigenface(U,v,X)


	i = 0
	mi = 0
	mv = 1e9
	for ok in O.T:
		L2 = np.linalg.norm(o-ok)
		if L2 < mv:
			mi = i
			mv = L2
		i += 1
	
	return mi
