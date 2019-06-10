import numpy as np
import cv2
import math

class MinutiaeExtractor:
	def __init__(self, fing_id, board):
		self.fing_id = fing_id
		self.board = board

	def extract(self, image, roi, block_size):
		binarized_image = self.binarize(image, roi, block_size)
		smoothed_image = self.smooth_binarized_image(binarized_image)
		fingerprint_skeleton = self.skeletonize(smoothed_image, roi, block_size)


	# Thining
	def binarize(self, image, roi, block_size):
		(width, height) = image.shape
		bins = range(255)
		histogram, _ = np.histogram(image, bins)
		binarized_image = self.board.create_blank(image.shape[0], image.shape[1])

		# print("histogram: {}\nTotal = {}".format(histogram, histogram_total))

		quarter_percentile, half_percientile = self.get_percentiles(histogram)

		# self.add_to_plot(histogram, [0,2])
		# print("p25 = {}; p50 = {}".format(quarter_percentile, half_percientile))

		for i in range(0, width):
			for j in range (0, height):
				# print (int(i), int(j))
				# print(width, height)
				if (roi[(i/block_size) - 1][(j/block_size) - 1] == 0):
					continue

				if (image[i][j] < quarter_percentile):
					binarized_image[i][j] = 0
				elif (image[i][j] > half_percientile):
					binarized_image[i][j] = 1
				else:
					binarized_image[i][j] = self.compare_mean(image, i, j)

		if self.fing_id.draw:
			self.board.add_to_plot(binarized_image, [1,2], 'binarized image')
		return binarized_image

	def compare_mean(self, image, i, j):
		(width, height) = image.shape
		cells = []
		block_start = -1
		block_end = 1
		for i in range(block_start, block_end):
			for j in range (block_start, block_end):
				if (i <= width and j<=height):
					cells.append((i,j))	
			block_pixels = [image[i - k][j - l] for k, l in cells]

		if (image[i][j] > np.mean(block_pixels)):
			return 255
		else:
			return 0

	def get_percentiles(self, histogram):
		histogram_total = sum(histogram)

		accumulator = 0
		quarter_percentile = 0
		half_percientile = 0
		for i in range(0, len(histogram)):
			accumulator += histogram[i]
			if (quarter_percentile == 0 and accumulator >= histogram_total * .25):
				quarter_percentile = i
			if (half_percientile == 0 and accumulator >= histogram_total * .5):
				half_percientile = i
				return quarter_percentile, half_percientile

	def smooth_binarized_image(self, binarized_image):
		# cells = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
		width, height = binarized_image.shape
		smoothed_image = binarized_image[:]# self.create_blank(width, height)
		for i in range(width):
			for j in range(height):
				# white_count, black_count = self.measure_noise(binarized_image, -2, 2, i, j)
				# if (white_count >= 18):
				# 	smoothed_image[i][j] = 1
				# elif (black_count >= 18):
				# 	smoothed_image[i][j] = 0
				# else:
				# 	smoothed_image[i][j] = binarized_image[i][j]

				very_smoothed_image = smoothed_image[:]
				white_count, black_count = self.measure_noise(smoothed_image, -1, 1, i, j)
				if (white_count >= 5):
					very_smoothed_image[i][j] = 1
				elif (black_count >= 5):
					very_smoothed_image[i][j] = 0
				else:
					very_smoothed_image[i][j] = smoothed_image[i][j]

		if self.fing_id.draw:
			self.board.add_to_plot(smoothed_image, [2,2], 'smoothed image')

		return smoothed_image


	def measure_noise(self, binarized_image, block_start, block_end, i, j):
		# cells = []
		# (width, height) = binarized_image.shape
		# for m in range(block_start, block_end):
		# 	for n in range (block_start, block_end):
		# 		if (i <= width and j<=height):
		# 			cells.append((m,n))

		# block_pixels = [binarized_image[i - k][j - l] for k, l in cells]

		black_count = 0
		white_count = 0
		
		for k in range(i - block_start, i + block_end):
			for l in range(j - block_start, j + block_end):
				if (k == i and l == j):
					continue
				if (binarized_image[i][j] == 1):
					white_count += 1
				if (binarized_image[i][j] == 0):
					black_count += 1

		return white_count, black_count

	def skeletonize(self, img, roi, block_size):
		# skel = cv2.distanceTransform(img, cv2.DIST_C, 3)
		# ret,img = cv2.threshold(img,127,255,0)

		width, height = img.shape
		size = np.size(img)
		skel = self.board.create_blank(img.shape[0], img.shape[1])
		# ret,img = cv2.threshold(img,127,255,0)
		element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

		done = False
		
		while( not done):
			eroded = cv2.erode(img,element)
			temp = cv2.dilate(eroded,element)
			temp = cv2.subtract(img,temp)
			skel = cv2.bitwise_or(skel,temp)
			img = eroded.copy()

			print(size)
			
			zeros = size - cv2.countNonZero(img)
			if zeros==size:
				done = True

		for i in range(0, width):
			for j in range (0, height):
				# print (k,l)
				if (roi[(i/block_size) - 1][(j/block_size) - 1] == 0):
					skel[i][j] = 0

		if self.fing_id.draw:
			self.board.add_to_plot(skel, [3, 2], 'fingerprint skeleton')
		
		return skel

	# Minutiae Extraction