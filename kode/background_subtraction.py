import numpy as np
from skimage.morphology import disk
from scipy.ndimage.filters import gaussian_filter
from skimage.morphology import ball
import scipy.ndimage as scim
from medpy.metric.image import mutual_information

def background_subtraction(h5_data, method, ball_radius = 50):
	global_min = h5_data.min().min()
	global_max = h5_data.max().max()
	grid_x = np.array(h5_data.index.get_level_values("grid_x"))
	grid_y = np.array(h5_data.index.get_level_values("grid_y"))

	img = np.zeros((grid_y.max() + 1, grid_x.max() + 1))

	for col in h5_data.columns:
		# Create 2D image
		img[(grid_y, grid_x)] = h5_data[col].values

		# Background subtraction based on strongly gaussian smoothed background image
		if method == "gaussian":
			# Create background image by strong gaussian smoothing
			img_bg = gaussian_filter(img, sigma=50)
			# Correct image by background image subtraction
			img_cor = img - img_bg

		# Background subtraction by adaption of rolling ball method
		if method == "rolling_ball":
			# Get the maximum intensity value in the current image
			img_max = img.max()
			# Create a ball as structuring element
			selem = ball(ball_radius)
			# Take only the upper half of the ball (tests on nematodes showed the same result for rescaling in 0-255 (img and ball) and r=50)
			# h = int((selem.shape[1]+1)/2)
			# selem = selem[:h,:,:].sum(axis=0)
			# Flatten the ball to a weighted disc by summation
			selem = selem.sum(axis=0)
			mi = selem.min()
			ma = selem.max()
			# Rescale disc into 0-img_max
			selem = (img_max * (selem - mi)) / (ma - mi)
			# Correct image by white-tophat calculation with the disc structuring element
			img_cor = scim.white_tophat(img, structure=selem)
			# mi = mutual_information(img, img_cor)
		h5_data[col] = img_cor[(grid_y, grid_x)]  # + mi
	# Scale processed image back into its original range to avoid too high or low numbers
	processed_min = h5_data.min().min()
	processed_max = h5_data.max().max()
	h5_data = (global_max - global_min) * ((h5_data - processed_min)/(processed_max - processed_min)) + global_min
	return h5_data
