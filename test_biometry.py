from fingerprint_lib import * 
fingerprint = FingerprintLib()

images = fingerprint.load_databases()
for image in images:
	cvImage = fingerprint.read_raw_image(image)
	fingerprint.add_to_plot(cvImage,[0,0])
	
	enhanced_img = fingerprint.enhance(cvImage)
	fingerprint.add_to_plot(enhanced_img, [1,0])

	block_size = 11
	gradient, gradient_image, alpha_x, alpha_y = fingerprint.compute_orientation(enhanced_img, block_size)
	fingerprint.add_to_plot(gradient, [2,0])
	fingerprint.add_to_plot(gradient_image, [3,0])

	valid_blocks = fingerprint.detect_roi(enhanced_img, block_size)
	fingerprint.add_to_plot(valid_blocks, [0,1])
	
	(smooth_direction, smooth_direction_image) = fingerprint.smooth_direction(enhanced_img, alpha_x, alpha_y, block_size, valid_blocks)

	fingerprint.add_to_plot(smooth_direction, [2,1])
	fingerprint.add_to_plot(smooth_direction_image, [3,1])

	fingerprint.plot()