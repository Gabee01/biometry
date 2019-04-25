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
			databaseImages = os.listdir(databasesPath + database)
			for image in databaseImages:
				images.append(databasesPath + database + image)

		return images

	def read_raw_image(self, image_path):
		image = np.empty(IMAGE_SIZE, np.uint8)
		image.data[:] = open(image_path).read()
		return image

	# Implement the fingerprint enhancement
	def enhance(self, image):
		(width, height) = image.shape
		enhanced_image = image[:]

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
		(width, height) = image.shape
		image = cv2.medianBlur(image,5)

		sobel = self.create_blank(width, height)
		alpha_x = [[0 for x in range(width)] for y in range(height)]
		alpha_y = [[0 for x in range(width)] for y in range(height)]

		for i in range(1,  width - 1):
			for j in range(1,  height - 1):
				z1 = image[i - 1,j - 1]
				z2 = image[i,j - 1]
				z3 = image[i + 1,j - 1]				
				z4 = image[i - 1, j]
				z5 = image[i,j]
				z6 = image[i + 1, j]
				z7 = image[i - 1,j + 1]				
				z8 = image[i, j + 1]
				z9 = image[i + 1,j + 1]

				gx = (z7 + 2*z8 + z9) - (z1 + 2*z2 + z3)
				gy = (z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)

				alpha_x[i][j] = math.pow(gx, 2) - math.pow(gy, 2)
				alpha_y[i][j] = 2 * gx * gy

		block_size = 10
		line_length = block_size * math.sqrt(2)

		average_x =	[[0 for x in range(width/block_size)] for y in range(height/block_size)]
		average_y = [[0 for x in range(width/block_size)] for y in range(height/block_size)]


		for i in range(1, width/block_size):
			for j in range(1, height/block_size):
				for k in range(i*block_size, (i+1) * block_size):
					for l in range(j*block_size, (j+1) * block_size):
						average_x[i][j] += alpha_x[k][l]
						average_y[i][j] += alpha_y[k][l]

		for i in range(1, width/block_size):
			for j in range(1, height/block_size):
				(x_zero, y_zero) = (i * block_size + block_size/2, j * block_size + block_size/2)
				
				average_x[i][j] = average_x[i][j]/pow(block_size, 2)
				average_y[i][j] = average_y[i][j]/pow(block_size, 2)
				gradient_direction = self.compute_block_angle(average_x[i][j], average_y[i][j])/2

				print(x_zero, line_length, math.cos(gradient_direction), gradient_direction)
				x = int(x_zero + line_length * math.cos(gradient_direction))
				y = int(y_zero + line_length * math.sin(gradient_direction))


				print ("(" + str(x_zero) + ", " + str(y_zero) + ") >> (" + str(x) + "," + str(y) + ")")
				cv2.line(image,(x_zero, y_zero),(x, y), (0,255,0), 1)

		return image

	def compute_block_angle(self, a_x, a_y):
		angle = 0.0
		if (a_x > 0):
			angle = np.arctan(a_y/a_x)
		if (a_x < 0 and a_y >= 0):
			angle = np.arctan(a_y/a_x) + np.pi
		if (a_x < 0 and a_y < 0):
			angle = np.arctan(a_y/a_x) - np.pi

		print (a_y, a_x, angle)

		return angle

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