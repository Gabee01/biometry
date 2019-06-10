import numpy as np
import math
import cv2

#Enhancement constants
ALPHA = 150
Y = 95

class TypeClassifier:
	def __init__(self, fing_id, board):
		self.fing_id = fing_id
		self.board = board

	def classify(self, cvImage, roi, block_size):
		average_ax, average_ay = self.compute_orientation(cvImage, block_size)
		
		(smooth_direction) = self.smooth_direction(cvImage, average_ax, average_ay, block_size, roi)

		poincare_image = self.compute_poincare(cvImage.shape, smooth_direction, roi, block_size)

	# def compare(self, fingerprint_type, fingerprint_annotation_type):
	# 	

	# Compute the Orientation Map
	def compute_orientation(self, image, block_size):
		image = cv2.medianBlur(image,5)

		alpha_x, alpha_y = self.compute_oritentation_gradient(image)

		average_x, average_y = self.compute_average_gradient(image, alpha_x, alpha_y, block_size)

		gradient_direction = self.compute_block_gradient(image, average_x, average_y, block_size)
		gradient = []
		if self.fing_id.draw:
			gradient, image = self.draw_gradient(image, gradient_direction, block_size)
			self.board.add_to_plot(gradient, [2,0], 'gradient')
			self.board.add_to_plot(image, [3,0], 'gradient image')

		return (average_x, average_y)


	def compute_block_gradient(self, image, average_x, average_y, block_size):
		(width, height) = image.shape
		gradient_direction = [[0 for x in range(width/block_size)] for y in range(height/block_size)]
		for i in range(0, width/block_size):
			for j in range(0, height/block_size):
				gradient_direction[i][j] = np.arctan2(average_y[i][j], average_x[i][j]) * .5 + np.pi/2#self.compute_block_angle(average_x[i][j], average_y[i][j])

		return (gradient_direction)

	def compute_average_gradient(self, image, alpha_x, alpha_y, block_size):
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

	def compute_oritentation_gradient(self, image):
		(width, height) = image.shape
		sobel = self.board.create_blank(width, height)

		gx = cv2.Sobel(image,cv2.CV_32F,1,0,ksize=3)
		gy = cv2.Sobel(image,cv2.CV_32F,0,1,ksize=3)

		alpha_x = gx ** 2 - gy ** 2
		alpha_y = 2 * gx * gy

		return (alpha_x, alpha_y)

	# Singular point detection (Poincare index)
	def smooth_direction(self, image, average_ax, average_ay, block_size, valid_blocks):
		smoothed_directions = self.compute_smoothed_directions(average_ax, average_ay, block_size, valid_blocks)

		if self.fing_id.draw:
			direction_ploted, smooth_direction_image = self.draw_gradient(image, smoothed_directions, block_size, valid_blocks)
			self.board.add_to_plot(smooth_direction_image, [1, 1], 'smooth directions')
			self.board.add_to_plot(direction_ploted, [2,1], 'directions ploted')
		return (smoothed_directions)

	def compute_smoothed_directions(self, alpha_x, alpha_y, block_size, valid_blocks):
		(width, height) = len(alpha_x), len(alpha_x[1])
		smoothed_blocks = [[0 for x in range(width)] for y in range(height)]
		blocks_offset = 1
		for k in range (0, width):
			for l in range (0, height):
				# print (k,l)
				if (valid_blocks[k][l] == 0):
					smoothed_blocks[k][l] = 0
					continue
				# print (k,l,"valid")
				center_block = (k + blocks_offset, l + blocks_offset)
				a = 0
				b = 0

				for m in range(center_block[0] - blocks_offset, center_block[0] + blocks_offset):
					for n in range (center_block[1] - blocks_offset, center_block[1] + blocks_offset):
						if ((m, n) != (center_block[0], center_block[1])):
							if (m > width) or (n > height):
								continue
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
		image_copy = np.copy(image)
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
				cv2.line(image_copy,(x_zero,y_zero), (x, y), (0,255,0), 2)
				cv2.line(gradient_image,(x_zero,y_zero), (x, y), (0,255,0), 2)
				
				# Draw both directions
				gradient_direction[i][j] = gradient_direction[i][j] + np.pi
				x = int(x_zero + line_length * math.cos(gradient_direction[i][j]))
				y = int(y_zero + line_length * math.sin(gradient_direction[i][j]))
				cv2.line(image_copy,(x_zero,y_zero), (x, y), (0,255,0), 2)
				cv2.line(gradient_image,(x_zero,y_zero), (x, y), (0,255,0), 2)

				gradient_direction[i][j] = gradient_direction[i][j] - np.pi

				# print('O = [{},{}], G = [{},{}], degrees = {}'.format(x_zero, y_zero, x, y, math.degrees(gradient_direction[i][j])))
		return (image_copy, gradient_image)
	# Poincare
	def compute_poincare(self, image_shape, angles, valid_blocks, block_size):
		(width, height) = image_shape

		singularities_image = self.board.create_blank(width, height)
		singularities_dictionary = {
			"loop": [],
			"delta": [],
			# "whorl": []
		}

		colors = {
			"loop" : (150, 0, 0), 
			"delta" : (0, 150, 0), 
			# "whorl": (0, 0, 150)
		}

		for i in range(1, len(angles) - 1):
			for j in range(1, len(angles[i]) - 1):
				if (valid_blocks[i][j] == 0):
					continue
				circle_size = 10
				singularity = self.poincare_index_at(i, j, angles)
				if singularity != "none":
					singularities_dictionary[singularity].append((i, j))
					# print (singularities_dictionary)

		self.remove_duplicates(singularities_dictionary, width, height)

		print(singularities_dictionary)

		if self.fing_id.draw:
			for singularity in singularities_dictionary:
				for (i, j) in singularities_dictionary[singularity]:
					cv2.circle(singularities_image,((j+1) * block_size, (i+1) * block_size), circle_size, colors[singularity], -1)
			self.board.add_to_plot(singularities_image, [0,2], 'poincare image')

		return singularities_image


	def remove_duplicates(self, singularities, width, height):
		center = (width/2, height/2)

		for singularity in singularities:
			# print(singularity)
			for (i, j) in singularities[singularity]:
				current_singularity = (i,j)
				for (k, l) in singularities[singularity]:
					if (i == k and j == l):
						continue
					other_singularity = (k, l)
					if (abs(self.get_distance(center, current_singularity) - self.get_distance(center, other_singularity)) < 11):
						singularities[singularity].remove((k,l))


	def get_distance(self, (x1, y1), (x2, y2)):
		return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))


	def poincare_index_at(self, i, j, angles):
		tolerance = 2
		cells = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
		deg_angles = [math.degrees(angles[i - k][j - l]) % 180 for k, l in cells]
		index = 0
		for k in range(0, 8):
			if abs(self.get_angle(deg_angles[k], deg_angles[k + 1])) > 90:
				deg_angles[k + 1] += 180
			index += self.get_angle(deg_angles[k], deg_angles[k + 1])

		if 180 - tolerance <= index and index <= 180 + tolerance:
			return "loop"
		if -180 - tolerance <= index and index <= -180 + tolerance:
			return "delta"
		# if 360 - tolerance <= index and index <= 360 + tolerance:
		# 	return "whorl"
		return "none"

	def get_angle(self, left, right):
		signum = lambda x: -1 if x < 0 else 1
		angle = left - right
		if abs(angle) > 180:
			angle = -1 * signum(angle) * (360 - abs(angle))
		return angle



