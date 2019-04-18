from biometry_lib import * 

images = loadDatabases()
for image in images:
	cvImage = readRawImage(image)
	addToPlotAndShow(cvImage)

	