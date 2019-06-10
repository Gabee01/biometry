import argparse
from loader import *
from board import *
from fingerprint_lib import *
from type_classifier import *
from minutiae_extractor import *

# fingerprint enhancement
def enhance(image):
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

# Region of interest detection
def detect_roi(image, block_size):
	width, height = image.shape
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
			block_ratio_distance = get_ratio(image_center, block_center, image_center_distance)

			if (is_valid(block_ratio_distance, mean[i][j], max_mean, std_dev[i][j], max_std_dev)):
				valid_blocks[i][j] = 1

	if fing_id.draw:
		board.add_to_plot(valid_blocks, [0,1], 'valid blocks (ROI)')

	return valid_blocks

# v = weight_mean (1-u) + weight_std_dev * o + w2
# weight_mean = 0.5; weight_std_dev = 0.5; w2 = (ratio of the distance to the center)
# u and o are normalized to be in [0,1]
# if v > 0.8, the block "is good"
def is_valid(ratio_distance, mean_block, max_mean, std_dev_block, max_std_dev):
	weight_mean = 0.5
	weight_std_dev = 0.5
	
	mean = mean_block/max_mean
	std_dev = std_dev_block/max_std_dev

	v = weight_mean * (1 - mean) + weight_std_dev * std_dev + ratio_distance * 1
	# print("mean_block/max_mean: {}/{} = {}, std_dev_block/max_std_dev: {}/{} = {}, ratio:{}, v:{}"
		# .format(mean_block, max_mean, mean, std_dev_block, max_std_dev, std_dev, ratio_distance, v))

	if v > 0.8:
		return True

def get_ratio(image_center, block_center, greatest_distance):
	block_distance = math.sqrt(math.pow(block_center[0] - image_center[0], 2) + math.pow(block_center[1] - image_center[1], 2))
	# print('block distance = {}'.format(block_distance))
	return 1 - block_distance/greatest_distance

arg_parser = argparse.ArgumentParser("fing-id")
arg_parser.add_argument("--draw", help="enables drawing images (impacts performance).", action='store_true')
arg_parser.add_argument("--type_classification", help="runs the type classification algorithm.", action='store_true')
arg_parser.add_argument("--min_extraction", help="runs the minutiae extraction algorithm.", action='store_true')
arg_parser.add_argument("--pattern_matching", help="executes the pattern matching for type and minutiae.", action='store_true')

fing_id = arg_parser.parse_args()

loader = Loader()
images = loader.load_databases()
images_annotations = loader.load_types()
board = Board()

# for images_annotations in images_annotations:
# 	try:
# 		# print(images_annotations)
# 		cvImage = loader.read_raw_image(images_annotations['imagePath'].split('/')[-1])

# 		if fing_id.type_classification:
# 			print("running type classification...")
# 			classifier = TypeClassifier(fing_id, board)
# 			fingerprint_type = classifier.classify(cvImage)
# 			# classifier.compare(fingerprint_type, fingerprint_annotation_type)
		
# 		if fing_id.min_extraction:
# 			print("running minutiae extraction...")
# 			extractor = MinutiaeExtractor(fing_id, board)
# 			fingerprint_minutiae = extractor.extract_minutiae(cvImage)

# 		# if fing_id.pattern_matching:
# 		# 	matcher = FingerprintPatternMatcher()
# 		# 	matcher.match(fingerprint_type, fingerprint_minutiae)

# 		if fing_id.draw:
# 			board.plot()
# 	except():
# 		print('error...')

for image in images:
	try:
		print(image)

		cvImage = loader.read_raw_image(image)

		if cvImage == []:
			continue

		if fing_id.draw:
			board.add_to_plot(cvImage,[0,0], 'original image')

		cvImage = enhance(cvImage)

		block_size = 11
		roi = detect_roi(cvImage, block_size)
		if fing_id.draw:
			board.add_to_plot(cvImage, [1,0], 'enhanced image')
		if fing_id.type_classification:
			classifier = TypeClassifier(fing_id, board)
			fingerprint_type = classifier.classify(cvImage, roi, block_size)
			# classifier.compare(fingerprint_type, fingerprint_annotation_type)
		
		if fing_id.min_extraction:
			extractor = MinutiaeExtractor(fing_id, board)
			fingerprint_minutiae = extractor.extract(cvImage, roi, block_size)

		# if fing_id.pattern_matching:
		# 	matcher = FingerprintPatternMatcher()
		# 	matcher.match(fingerprint_type, fingerprint_minutiae)

		if fing_id.draw:
			board.plot()
	except():
		print('error...')