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

def mat_from_img_path(img_path):
	img = np.asarray( cv.LoadImageM(img_path , cv.CV_LOAD_IMAGE_GRAYSCALE) )
	return np.reshape( img[:,:], (reduce(operator.mul, img.shape), 1))
	
def compute_eigenface(U,v,X):
	p = np.subtract(v,X)
	eigf = np.dot(U,p)
	return eigf
# =====================================
#
# Training
#
# =====================================
def train(db_path="",):

	print "Training: loading from database ",db_path
	file_suffix_matcher = re.compile(".jpg|.jpeg|.jpe|.pgm|.ppm|.pbm|.png|.tiff|.tif")
	files = os.listdir(db_path)

	# Load files as cvMats
	images = [ np.asarray(cv.LoadImageM(db_path+file_name, cv.CV_LOAD_IMAGE_GRAYSCALE)) for file_name in files ]
		
	# Reshape and stack images as columns of a huge matrix
	images = np.asarray(map(lambda M: np.reshape(M[:,:],reduce(operator.mul, M.shape), 1), images))
	A = np.matrix(images.T)

	print "Training: computing eigenface basis"
	X = np.array(np.mean(A,axis=1))
	B = np.subtract(A,X)
	D,V = np.linalg.eig( np.dot(B.T,B) )

	print "Training: sorting by descending eigen value"
	ind = filter(lambda x: x>0.0, D.argsort()[::-1])
	D = D[ind]
	V = V[:,ind]

	print "Training: computing eigenfaces"
	D = np.sqrt(1.0/D)
	D = np.diag(D)
	U = np.dot(B, np.dot(V,D))

	A = np.asarray([compute_eigenface(U.T,v,X) for v in A])

	return A, U.T, X


# =====================================
# Recognition
#
# X is the is the average face
# U is the eigenface basis.
# 
# =====================================
def lookup(A,mat,U,X):
	
	mat = np.reshape(mat, (reduce(operator.mul, mat.shape), 1))

	v = compute_eigenface(mat)

	p = np.subtract(v,X)
	
	print "p",p.shape
	# Projection onto face space
	eigf = np.dot(U, p)
	
	print X.shape, eigf.shape
	e = np.subtract(X,eigf)
	E = np.subtract(X,U)
	
	d = abs(np.sum(np.square(np.subtract(eigf,E))/np.square(np.std(np.subtract(eigf,E))),axis=0))
	
	print list(d).index(min(d))	
 
