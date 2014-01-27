Face Recognition - Simple face recognition based on eigenfaces
==============================================================

face_recognition is a self-contained implementation of the Eigenface-method in Python using OpenCV and NumPy.
This repository provides the core-library and some applications for testing. 

* Website: [www.indrome.com](www.indrome.com)

Dependencies
------------

Python 2.X (tested 2.7.6) 
Numpy 1.8 (tested 1.8.0)

Example: testing
----------------

If you have the correct dependencies installed you can use the library file as such:

```python
import eigenface

# load all reference faces
files = file_list_from_path("path-to-face-files")

# O is a set of columns corresponding to the eigenfaces of files
# UT is the eigenspace-basis
# X is the mean faces of files

O,UT,X = eigenface.train(files)

# load another face
vec = vec_from_img_path("path-to-face.jpg")

eigenface.recognize(O,U,vec,X) # return the column-index in O which was recognized as the input vec
```
