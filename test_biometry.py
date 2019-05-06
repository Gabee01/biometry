from fingerprint_lib import * 

plot_lines = 5
plot_cols = 2
fingerprint = FingerprintLib()

images = fingerprint.load_databases()
for image in images:
	cvImage = fingerprint.read_raw_image(image)
	fingerprint.add_to_plot(cvImage,[0,0])
	
	enhanced_img = fingerprint.enhance(cvImage)
	fingerprint.add_to_plot(enhanced_img, [1,0])

	block_size = 10
	gradient, gradient_image = fingerprint.compute_orientation(enhanced_img, block_size)
	fingerprint.add_to_plot(gradient, [2,0])
	fingerprint.add_to_plot(gradient_image, [3,0])

	roi_image = fingerprint.detect_roi(gradient, block_size)
	fingerprint.add_to_plot(roi_image, [0,1])

	fingerprint.plot()