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

	oriented_image = fingerprint.compute_orientation(enhanced_img)
	fingerprint.add_to_plot(oriented_image, [2,0])

	fingerprint.plot()