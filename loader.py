import os
import json
import numpy as np

class Loader():
	def __init__(self):
		self.raw_image_size = (300, 300)
		self.databasesPath = os.getcwd() + "/databases/"
		self.databaseDir = "Rindex28/"
		self.typesDir = "Rindex28-type/"	

	# Code to load the databases
	def load_databases(self):
		images = []
		images_path = self.databasesPath + self.databaseDir

		databaseImages = os.listdir(images_path)
		for image in databaseImages:
			images.append(image)

		return images

	
	def load_types(self):
		types = []
		types_path = self.databasesPath + self.typesDir

		types_files = os.listdir(types_path)
		for file in types_files:
			if '.lif' in file:
				file_path = types_path + file

				# print("openning file {}".format(file_path))
				with open(file_path, 'r') as type_file:
					file_type_annotation = json.load(type_file)
					types.append(file_type_annotation)

		# print("read {} types: printing first:\n{}".format(len(types), types[0]))
		return types


	def read_raw_image(self, image_name):
		images_path = self.databasesPath + self.databaseDir
		image = np.empty(self.raw_image_size, np.uint8)
		if '.pgm' in image_name:
			image_name = image_name.replace('.pgm', '.raw')
		if '.raw' not in image_name:
			return
		image.data[:] = open(images_path + image_name).read()
		return image