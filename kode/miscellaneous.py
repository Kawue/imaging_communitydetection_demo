from __future__ import print_function, division

import numpy as np
import pandas as pd
import os

def write_dataframe_to_disc(dataframe, out_path, dataset_name):
	h5_store_path = os.path.join(out_path, dataset_name + '.h5')
	save_name_frame = 'msi_frame_' + dataset_name
	with pd.HDFStore(h5_store_path, complib='blosc', complevel=9) as store:
		store[save_name_frame] = dataframe


def gaussian_2d_images(param_set):
	if param_set == "motifs":
		# Gaussian Parameter Set Motifs
		x_ax,y_ax = np.mgrid[0:301,0:401]
		# (Single - S), (Double - D), (Triple - T), (Nested - N), (Ring - R), (Wall - W), (Noise - X)
		nb_images = 72
		nb_noise_images = 5
		np.random.seed(0)
		rand_1 = np.random.uniform(-1, 1, nb_images + nb_noise_images)
		np.random.seed(10)
		rand_2 = np.random.uniform(-1, 1, nb_images + nb_noise_images)
		np.random.seed(100)
		rand_3 = np.random.uniform(-2, 2, nb_images + nb_noise_images)
		np.random.seed(1000)
		rand_4 = np.random.uniform(-2, 2, nb_images + nb_noise_images)

		params = {
			"center_x": np.array([25, 25, 25,

								  130, 130, 130,
								  165, 165, 165,

								  60, 60, 60,
								  70, 70, 70,
								  90, 90, 90,

								  80, 80, 80,
								  80, 80, 80,
								  80, 80, 80,

								  235, 235, 235,
								  205, 205, 205,
								  235, 235, 235,
								  265, 265, 265,
								  235, 235, 235,

								  250, 250, 250,
								  250, 250, 250,
								  250, 250, 250,
								  250, 250, 250,
								  250, 250, 250,
								  270, 270, 270,
								  270, 270, 270,
								  270, 270, 270,
								  270, 270, 270,
								  270, 270, 270,

								  0, 0, 0, 0, 0]) + rand_1,
			"center_y": np.array([25, 25, 25,

								  40, 40, 40,
								  40, 40, 40,

								  130, 130, 130,
								  175, 175, 175,
								  150, 150, 150,

								  320, 320, 320,
								  320, 320, 320,
								  320, 320, 320,

								  340, 340, 340,
								  340, 340, 340,
								  370, 370, 370,
								  340, 340, 340,
								  310, 310, 310,

								  65, 65, 65,
								  90, 90, 90,
								  115, 115, 115,
								  140, 140, 140,
								  165, 165, 165,
								  70, 70, 70,
								  95, 95, 95,
								  120, 120, 120,
								  145, 145, 145,
								  170, 170, 170,

								  0, 0, 0, 0, 0]) + rand_2,
			"sigma_x": np.array([5, 5, 5,

								 10, 10, 10,
								 15, 15, 15,

								 25, 25, 25,
								 25, 25, 25,
								 25, 25, 25,

								 5, 5, 5,
								 25, 25, 25,
								 50, 50, 50,

								 30, 30, 30,
								 10, 10, 10,
								 10, 10, 10,
								 10, 10, 10,
								 10, 10, 10,

								 10, 10, 10,
								 10, 10, 10,
								 10, 10, 10,
								 10, 10, 10,
								 10, 10, 10,
								 10, 10, 10,
								 10, 10, 10,
								 10, 10, 10,
								 10, 10, 10,
								 10, 10, 10,

								 0, 0, 0, 0, 0]) + rand_3,
			"sigma_y": np.array([5, 5, 5,

								 10, 10, 10,
								 15, 15, 15,

								 25, 25, 25,
								 25, 25, 25,
								 25, 25, 25,

								 5, 5, 5,
								 25, 25, 25,
								 50, 50, 50,

								 30, 30, 30,
								 10, 10, 10,
								 10, 10, 10,
								 10, 10, 10,
								 10, 10, 10,

								 10, 10, 10,
								 10, 10, 10,
								 10, 10, 10,
								 10, 10, 10,
								 10, 10, 10,
								 10, 10, 10,
								 10, 10, 10,
								 10, 10, 10,
								 10, 10, 10,
								 10, 10, 10,

								 0, 0, 0, 0, 0]) + rand_4,
			"theta": np.array([0, 0, 0,

							   0, 0, 0,
							   0, 0, 0,

							   0, 0, 0,
							   0, 0, 0,
							   0, 0, 0,

							   0, 0, 0,
							   0, 0, 0,
							   0, 0, 0,

							   0, 0, 0,
							   0, 0, 0,
							   0, 0, 0,
							   0, 0, 0,
							   0, 0, 0,

							   0, 0, 0,
							   0, 0, 0,
							   0, 0, 0,
							   0, 0, 0,
							   0, 0, 0,
							   0, 0, 0,
							   0, 0, 0,
							   0, 0, 0,
							   0, 0, 0,
							   0, 0, 0,

							   0, 0, 0, 0, 0]) + np.random.uniform(-5,5,nb_images + nb_noise_images),
			"amplitude": np.random.random(nb_images) + 5
		}
	elif param_set == "circles":
		# Gaussian Parameter Set Circles
		x_ax, y_ax = np.mgrid[0:190, 0:205]
		nb_images = 9
		nb_noise_images = 0
		np.random.seed(0)
		rand_1 = np.random.uniform(-5, 5, nb_images + nb_noise_images)
		np.random.seed(10)
		rand_2 = np.random.uniform(-5, 5, nb_images + nb_noise_images)
		np.random.seed(100)
		rand_3 = np.random.uniform(-5, 5, nb_images + nb_noise_images)
		np.random.seed(1000)
		rand_4 = np.random.uniform(-5, 5, nb_images + nb_noise_images)
		params = {
			"center_x": np.array([80, 80, 80, 90, 90, 90, 110, 110, 110]) + rand_1,
			"center_y": np.array([80, 80, 80, 125, 125, 125, 100, 100, 100]) + rand_2,
			"sigma_x": np.array([25, 25, 25, 25, 25, 25, 25, 25, 25]) + rand_3,
			"sigma_y": np.array([25, 25, 25, 25, 25, 25, 25, 25, 25]) + rand_4,
			"theta": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]) + np.random.uniform(-5, 5, nb_images + nb_noise_images),
			"amplitude": np.random.random(nb_images) + 5
		}
	else:
		raise ValueError("Wrong value for param_set in gaussian_2D_images method.")

	noiselevel = 0
	
	def gaussian(x,y, amplitude, center_x, center_y, sigma_x, sigma_y, theta):
		a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
		b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
		c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
		#g = offset + amplitude*np.exp(-(a*((x-center_x)**2) + 2*b*(x-center_x)*(y-center_y) + c*((y-center_y)**2)))
		g = amplitude * np.exp(-(a * ((x - center_x) ** 2) + 2 * b * (x - center_x) * (y - center_y) + c * ((y - center_y) ** 2)))
		return g.ravel()

	# Create Images
	images = []
	for i in range(nb_images + nb_noise_images):
		if i < nb_images:
			data = gaussian(x_ax, y_ax, params["amplitude"][i], params["center_x"][i], params["center_y"][i], params["sigma_x"][i], params["sigma_y"][i], params["theta"][i])
		else:
			data = np.zeros(x_ax.shape).ravel()
			data += np.random.uniform(0,1,data.shape)
		data += noiselevel * np.random.random(data.shape)
		data = data.reshape(x_ax.shape)
		images.append(data)

	# Maximum x and y pixel from image size
	y_max, x_max = data.shape
	# List of (y,x)-pixel tuples
	pixel_pos = np.array([(y,x) for y in range(y_max) for x in range(x_max)])
	y_pixels = pixel_pos[:,0]
	x_pixels = pixel_pos[:,1]

	# Use enumeration as identifier/name
	circles = [float(x) for x in range(nb_images + nb_noise_images)]

	# Create array with image intensity values, indexed by (x,y)-pixel tuples
	intensity_values = np.array([img[y_pixels, x_pixels] for img in images]).T

	# Create array with (x,y)-pixel tuples
	pos = np.array([x_pixels, y_pixels])

	# Create pixel-index of the image
	pos_index = pd.MultiIndex.from_arrays(pos, names=("grid_x", "grid_y"))

	# Create DataFrame
	df = pd.DataFrame(intensity_values, index=pos_index, columns=circles)
	return df
	

def gaussian(image, amplitude, center_x, center_y, sigma_x, sigma_y):
	x, y = np.mgrid[0:image.shape[0], 0:image.shape[1]]
	# Returns a gaussian function with the given parameters
	return amplitude * np.exp(-(((x - center_x) ** 2 / (2 * sigma_x ** 2)) + ((y - center_y) ** 2 / (2 * sigma_y ** 2))))