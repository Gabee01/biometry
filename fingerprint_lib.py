import os
import io
import math
import cv2
from matplotlib import pyplot as plt
import numpy as np

IMAGE_SIZE = (300, 300)
PLOT_LINES = 4
PLOT_COLS = 2

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

		mean = np.mean(image)
		variance = np.var(image)
		for i in range(0,  width):
			for j in range(0,  height):
				if image[i, j] < 2:
					enhanced_image[i, j] = 255
				else:
					s = math.sqrt(variance)
					if (Y < s):
						enhanced_image[i, j] = ALPHA + Y * ((image[i, j] - mean)/s)
					else:
						enhanced_image[i, j] = ALPHA + Y * ((image[i, j] - mean)/Y)

		return enhanced_image

	# Compute the Orientation Map
	def compute_orientation(self, image, block_size):
		image = cv2.medianBlur(image,5)

		alpha_x, alpha_y = self.compute_alpha(image)

		average_x, average_y = self.compute_average(image, alpha_x, alpha_y, block_size)

		gradient_direction = self.compute_gradient(image, average_x, average_y, block_size)
		gradient, image = self.draw_gradient(image, gradient_direction, block_size)
		return (gradient, image, average_x, average_y)


	def compute_gradient(self, image, average_x, average_y, block_size):
		(width, height) = image.shape
		gradient_direction = [[0 for x in range(width/block_size)] for y in range(height/block_size)]
		for i in range(0, width/block_size):
			for j in range(0, height/block_size):
				gradient_direction[i][j] = np.arctan2(average_y[i][j], average_x[i][j]) * .5 + np.pi/2#self.compute_block_angle(average_x[i][j], average_y[i][j])

		return (gradient_direction)

	def compute_average(self, image, alpha_x, alpha_y, block_size):
		(width, height) = image.shape

		average_x =	[[0 for x in range(width/block_size)] for y in range(height/block_size)]
		average_y = [[0 for x in range(width/block_size)] for y in range(height/block_size)]

		for i in range(0, width/block_size):
			for j in range(0, height/block_size):
				for k in range(i*block_size, (i+1) * block_size):
					for l in range(j*block_size, (j+1) * block_size):
						average_x[i][j] += alpha_x[k][l]
						average_y[i][j] += alpha_y[k][l]
				
				average_x[i][j] = average_x[i][j]/pow(block_size, 2)
				average_y[i][j] = average_y[i][j]/pow(block_size, 2)

		return average_x, average_y

	def compute_alpha(self, image):
		(width, height) = image.shape
		sobel = self.create_blank(width, height)

		gx = cv2.Sobel(image,cv2.CV_32F,1,0,ksize=3)
		gy = cv2.Sobel(image,cv2.CV_32F,0,1,ksize=3)

		alpha_x = gx ** 2 - gy ** 2
		alpha_y = 2 * gx * gy

		return (alpha_x, alpha_y)

	# Load the Fingeprint type annotation
	# Region of interest detection
	def detect_roi(self, image, block_size):
		(width, height) = image.shape
		mean = [[0 for x in range(width/block_size)] for y in range(height/block_size)]
		std_dev = [[0 for x in range(width/block_size)] for y in range(height/block_size)]
		max_mean = 0
		max_std_dev = 0

		for i in range (0, width/block_size):
			for j in range (0, height/block_size):
				block = []
				block_zero = ((i * block_size), (j * block_size))
				block_end = (block_zero[0] + block_size, block_zero[1] + block_size)

				# block = image[[block_zero[0], block_end[0]], :][:,[block_zero[1], block_end[1]]]
				for k in range (block_zero[0], block_end[0]):
					for l in range (block_zero[1], block_end[1]):
						block.append(image[k][l])

				mean[i][j] = np.mean(block)
				std_dev[i][j] = np.std(block)

				if (mean[i][j] > max_mean):
					max_mean = mean[i][j]
				if (std_dev[i][j] > max_std_dev):
					max_std_dev = std_dev[i][j]

		image_center = (width/2, height/2)
		image_center_distance = image.shape[0] * math.sqrt(2) / 2
		valid_blocks = [[0 for x in range(width/block_size)] for y in range(height/block_size)]
		for i in range(0, width/block_size):
			for j in range(0, height/block_size):
				block_zero = ((i * block_size), (j * block_size))
				block_end = (block_zero[0] + block_size, block_zero[1] + block_size)

				block_center = ((block_zero[0] + block_size/2), (block_zero[1] + block_size/2))
				block_ratio_distance = self.get_ratio(image_center, block_center, image_center_distance)

				if (self.is_valid(block_ratio_distance, mean[i][j], max_mean, std_dev[i][j], max_std_dev)):
					valid_blocks[i][j] = 1
		return valid_blocks

	# v = weight_mean (1-u) + weight_std_dev * o + w2
	# weight_mean = 0.5; weight_std_dev = 0.5; w2 = (ratio of the distance to the center)
	# u and o are normalized to be in [0,1]
	# if v > 0.8, the block "is good"
	def is_valid(self, ratio_distance, mean_block, max_mean, std_dev_block, max_std_dev):
		weight_mean = 0.5
		weight_std_dev = 0.5
		
		mean = mean_block/max_mean
		std_dev = std_dev_block/max_std_dev

		v = weight_mean * (1 - mean) + weight_std_dev * std_dev + ratio_distance * 1
		# print("mean_block/max_mean: {}/{} = {}, std_dev_block/max_std_dev: {}/{} = {}, ratio:{}, v:{}"
			# .format(mean_block, max_mean, mean, std_dev_block, max_std_dev, std_dev, ratio_distance, v))

		if v > 0.8:
			return True

	def get_ratio(self, image_center, block_center, greatest_distance):
		block_distance = math.sqrt(math.pow(block_center[0] - image_center[0], 2) + math.pow(block_center[1] - image_center[1], 2))
		# print('block distance = {}'.format(block_distance))
		return 1 - block_distance/greatest_distance
	# Singular point detection (Poincare index)
	def smooth_direction(self, image, average_ax, average_ay, block_size, valid_blocks):
		smoothed_directions = self.compute_smoothed_directions(average_ax, average_ay, block_size, valid_blocks)
		return (smoothed_directions, self.draw_gradient(image, smoothed_directions, block_size, valid_blocks))

	def compute_smoothed_directions(self, alpha_x, alpha_y, block_size, valid_blocks):
		(width, height) = len(alpha_x), len(alpha_x[1])
		smoothed_blocks = [[0 for x in range(width)] for y in range(height)]
		blocks_offset = 1
		for k in range (0, width):
			for l in range (0, height):
				# print (k,l)
				if (valid_blocks[k][l] == 0):
					continue
				# print (k,l,"valid")
				center_block = (k + blocks_offset, l + blocks_offset)
				a = 0
				b = 0

				for m in range(center_block[0] - blocks_offset, center_block[0] + blocks_offset):
					for n in range (center_block[1] - blocks_offset, center_block[1] + blocks_offset):
						if ((m, n) != (center_block[0], center_block[1])):
							a += alpha_x[m][n] 
							b += alpha_y[m][n]

				a += 2 * alpha_x[center_block[0]][center_block[1]]
				b += 2 * alpha_y[center_block[0]][center_block[1]]
				# print ("[{},{}] - b = {}; a = {}; b/a = {}".format(m, n, b, a, b/a))
				smoothed_blocks[k][l] = np.arctan2(b, a)/2 + np.pi/2
		return smoothed_blocks

	def draw_gradient(self, image, gradient_direction, block_size, roi = []):
		(width, height) = image.shape
		gradient_image = np.empty(image.shape, np.uint8)
		gradient_oposite = np.empty(image.shape, np.uint8)
		gradient_image.fill(255)
		line_length = block_size / 2 + 1

		for i in range(0, width/block_size):
			for j in range(0, height/block_size):
				if (roi != [] and roi[i][j] == 0):
					continue
				block_center = (i * block_size + block_size/2, j * block_size+block_size/2)
				# print('graditent[{}][{}] = arctan = {} rad'.format(i, j, gradient_direction[i][j]))
				# (x_zero, y_zero) = (i * block_size, j * block_size + block_size)
				(y_zero, x_zero) = block_center

				x = int(x_zero + line_length * math.cos(gradient_direction[i][j]))
				y = int(y_zero + line_length * math.sin(gradient_direction[i][j]))
				cv2.line(image,(x_zero,y_zero), (x, y), (0,255,0), 2)
				cv2.line(gradient_image,(x_zero,y_zero), (x, y), (0,255,0), 2)
				
				# Draw both directions
				gradient_direction[i][j] = gradient_direction[i][j] + np.pi
				x = int(x_zero + line_length * math.cos(gradient_direction[i][j]))
				y = int(y_zero + line_length * math.sin(gradient_direction[i][j]))
				cv2.line(image,(x_zero,y_zero), (x, y), (0,255,0), 2)
				cv2.line(gradient_image,(x_zero,y_zero), (x, y), (0,255,0), 2)

				gradient_direction[i][j] = gradient_direction[i][j] - np.pi

				# print('O = [{},{}], G = [{},{}], degrees = {}'.format(x_zero, y_zero, x, y, math.degrees(gradient_direction[i][j])))
		return (image, gradient_image)
	# Poincare
	def compute_poncare(self, image, gradient, valid_blocks, block_size):
		(width, height) = image.shape
		for j in range (1, width/block_size - 1):
			for i in range (1, height/block_size - 1):
				# print (k,l)
				# if (valid_blocks[i][j] == 0):
					# continue
				# print (k,l,"valid")
				center_block = (i, j)

				p = []

				p.append(gradient[center_block[0] - 1][center_block[1] - 1])
				p.append(gradient[center_block[0] - 1][center_block[1]])
				p.append(gradient[center_block[0] - 1][center_block[1] + 1])
				p.append(gradient[center_block[0]][center_block[1] + 1])
				p.append(gradient[center_block[0] + 1][center_block[1] + 1])
				p.append(gradient[center_block[0] + 1][center_block[1]])
				p.append(gradient[center_block[0] + 1][center_block[1] - 1])
				p.append(gradient[center_block[0]][center_block[1] - 1])

				# angle = (p1 - p2) + (p3 - p2) + (p4 - p3) + (p5 - p4) + (p6 - p5) + (p7 - p5) + (p8 - p7) + (p1 - p8)

				pi = 1/np.pi * self.orientation_sum(p)

				print ("poincare index = {}".format(pi))
				if (math.degrees(pi) in range(175, 185)):
					# print ("poincare index = {}".format(pi))
					# print ("{}, {} = {} rad".format(i, j, pi))
					cv2.circle(image,center_block, 5, (0,0,255), -1)
				if (math.degrees(pi) in range(-185, -175)):
					# print ("poincare index = {}".format(pi))
					# print ("{}, {} = {} rad".format(i, j, pi))
					cv2.circle(image,center_block, 5, (0,0,255), -1)
		return image

	def orientation_sum(self, p):
		sum = 0

		for i in range(0, len(p) - 1):
			j = (i + 1) % 8
			sum += p[j] - p[i]
			print("sum = {}".format(sum))
		return sum

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