import os
import io
import math
import cv2
from matplotlib import pyplot as plt
import numpy as np

IMAGE_SIZE = (300, 300)
PLOT_LINES = 4
PLOT_COLS = 4

#Enhancement constants
ALPHA = 150
Y = 95

class FingerprintLib:
	def __init__ (self):
		self._fig, self._aplt = plt.subplots(PLOT_LINES, PLOT_COLS)

	# Code to load the databases
	def load_databases(self):
		databasesPath = os.getcwd() + "/databases/"
		databasesList = ["Lindex101/", "Rindex28/"]
		rindexTypeDir = "Rindex28-type/"

		images = []

		for database in databasesList:
			for image in os.listdir(databasesPath + database):
				images.append(databasesPath + database + image)

		return images

	def read_raw_image(self, image_path):
		image = np.empty(IMAGE_SIZE, np.uint8)
		image.data[:] = open(image_path).read()
		return image

	# Implement the fingerprint enhancement
	def enhance(self, image):
		(width, height) = image.shape
		enhanced_image = self.create_blank(width, height)

		for i in range(0,  width):
			for j in range(0,  height):
				s = math.sqrt(np.var(image))
				mean = np.mean(image)
				if (Y < s):
					enhanced_image[i, j] = ALPHA + Y * ((image[i, j] - mean)/s)
				else:
					enhanced_image[i, j] = ALPHA + Y * ((image[i, j] - mean)/Y)

		return enhanced_image

	# Compute the Orientation Map
	def compute_orientation(self, image):
		(width, height, _) = image.shape
		image = cv2.medianBlur(image,5)

		sobel = self.create_blank(width, height)

		for i in range(0,  width):
			for j in range(0,  height):
				
				z1 = 0
				z2 = 0
				z3 = 0
				z4 = 0
				z5 = 0
				z6 = 0
				z7 = 0
				z8 = 0
				z9 = 0

				z5 = image[i,j]

				if (i > 0):
					z4 = image[i - 1, j]
					if (j > 0):
						z1 = image[i - 1,j - 1]

				if (j > 0):
					z2 = image[i,j - 1]
					if (i < width - 1):
						z3 = image[i + 1,j - 1]

				if (j < height - 1):
					z8 = image[i, j + 1]
					if (i > 0):
						z7 = image[i - 1,j + 1]

				if (i < width - 1):
					z6 = image[i + 1, j]
					if (j < height - 1):
						z9 = image[i + 1,j + 1]

				gx = (z7 + 2*z8 + z9) - (z1 + 2*z2 + z3)
				gy = (z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)

			ro = math.sqrt(math.pow(gx, 2) + math.pow(gy, 2))
			theta = math.pow(math.atan(gy/gx), -1)

			print ("ro: " + ro)
			print ("theta: " + theta)
		return image
	# Load the Fingeprint type annotation
	# Region of interest detection
	# Singular point detection (Poincare index)
	# Fingerprint Type Classification
	# Thining
	# Minutiae Extraction
	# Pattern Matching


	#General helpers
	def plot(self):
		plt.pause(15)
		plt.close()
		self._fig, self._aplt = plt.subplots(PLOT_LINES, PLOT_COLS)

	def add_to_plot(self, image, positionToPlot):
		self._aplt[positionToPlot[0], positionToPlot[1]].imshow(image, cmap='Greys_r')

	def create_blank(self, width, height):
		blank_image = np.zeros((height, width, 3), np.uint8)
		return blank_image