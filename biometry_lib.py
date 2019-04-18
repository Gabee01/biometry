import os
import io
## Mathmatics libraries
import numpy as np
import math

## Image Processing libraries
import skimage

import scipy.misc as misc

import rawpy
# import imageio
from PIL import Image

## Visual and plotting libraries
import matplotlib.pyplot as plt

from cv_utils import *

hl, = plt.plot([], [])

IMAGE_SIZE = (300, 300)

# Code to load the databases
def loadDatabases():
	databasesPath = os.getcwd() + "/databases/"
	databasesList = ["Lindex101/", "Rindex28/"]
	rindexTypeDir = "Rindex28-type/"

	images = []

	for database in databasesList:
		for image in os.listdir(databasesPath + database):
			images.append(databasesPath + database + image)

	return images

def readRawImage(image_path):
	imageString = open(image_path).read()

	Image.fromstring('F', IMAGE_SIZE, imageString, 'raw', 'F;16')

	# imageFile = open(image_path, 'rb')

	# imageBytes = io.BytesIO(imageFile.read())
	# image = Image.open(imageBytes)

	image.show()

	plt.pause(50)

def addToPlotAndShow(image):
    hl.set_xdata(numpy.append(hl.get_xdata(), new_data))
    hl.set_ydata(numpy.append(hl.get_ydata(), new_data))
    plt.draw()
    plt.pause(3)

# Implement the fingerprint enhancement
# Compute the Orientation Map
# Load the Fingeprint type annotation
# Region of interest detection
# Singular point detection (Poincare index)
# Fingerprint Type Classification
# Thining
# Minutiae Extraction
# Pattern Matching
