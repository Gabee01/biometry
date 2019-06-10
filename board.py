from matplotlib import pyplot as plt
import numpy as np

PLOT_LINES = 4
PLOT_COLS = 3

class Board:
	def __init__(self):
		self._fig, self._aplt = plt.subplots(PLOT_LINES, PLOT_COLS)

	#General helpers
	def plot(self):
		plt.tight_layout()
		# plt.pause(0.001)
		plt.pause(15)
		plt.close()
		self._fig, self._aplt = plt.subplots(PLOT_LINES, PLOT_COLS)

	def add_to_plot(self, image, positionToPlot, title):
		plot = self._aplt[positionToPlot[0], positionToPlot[1]]
		plot.set_title(title)
		plot.imshow(image, cmap='Greys_r')


	def rgb2gray(self, rgb):
		return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


	def create_blank(self, width, height):
		blank_image = np.zeros((height, width, 3), np.uint8)

		return self.rgb2gray(blank_image)