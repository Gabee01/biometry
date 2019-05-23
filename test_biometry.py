from fingerprint_lib import * 
fingerprint = FingerprintLib()

images = fingerprint.load_databases()
for image in images:
	cvImage = fingerprint.read_raw_image(image)
	fingerprint.add_to_plot(cvImage,[0,0], 'original image')
	
	enhanced_img = fingerprint.enhance(cvImage)
	fingerprint.add_to_plot(enhanced_img, [1,0], 'enhanced image')

	block_size = 11
	gradient, gradient_image, average_ax, average_ay = fingerprint.compute_orientation(enhanced_img, block_size)
	fingerprint.add_to_plot(gradient, [2,0], 'gradient')
	fingerprint.add_to_plot(gradient_image, [3,0], 'gradient image')

	valid_blocks = fingerprint.detect_roi(enhanced_img, block_size)
	fingerprint.add_to_plot(valid_blocks, [0,1], 'valid blocks (ROI)')
	
	(smooth_direction, (direction_ploted, smooth_direction_image)) = fingerprint.smooth_direction(enhanced_img, average_ax, average_ay, block_size, valid_blocks)

	fingerprint.add_to_plot(smooth_direction, [1, 1], 'smooth directions')
	fingerprint.add_to_plot(direction_ploted, [2,1], 'directions ploted')
	fingerprint.add_to_plot(smooth_direction_image, [3,1], 'smooth directions image')

	poincare_image = fingerprint.compute_poincare(direction_ploted, smooth_direction, valid_blocks, block_size)
	fingerprint.add_to_plot(poincare_image, [0,2], 'poincare image')
	
	binarized_image = fingerprint.binarize(enhanced_img)
	fingerprint.add_to_plot(enhanced_img, [1,2], 'binarized image')

	smoothed_image = fingerprint.smooth_binarized_image(binarized_image)
	fingerprint.add_to_plot(smoothed_image, [2,2], 'smoothed image')

	# fingerprint_skeleton = fingerprint.skeletonize(smoothed_image)
	# fingerprint.add_to_plot(fingerprint_skeleton, [3, 2], 'fingerprint skeleton')	

	fingerprint.plot()