from __future__ import print_function, division

from builtins import ValueError
import pandas as pd
import skimage.filters as skif
import numpy as np
import math
from scipy.ndimage import filters
from skimage import restoration as res


# Reduce original dataframe by keeping only the indices passed by the list parameter
def reduce_dataframe(dataframe, keep_indices_list):
	reduced_dataframe = pd.DataFrame()
	for col_idx, col_name in enumerate(dataframe.columns):
		if col_idx in keep_indices_list:
			reduced_dataframe[col_name] = dataframe[col_name]
	return reduced_dataframe


# Threshold data with thresholding method while binarising
def thresholding_binarise(dataframe):
	# Compute threshold with thresholding method for each column
	thresholds = dataframe.apply(lambda x: threshold(x), axis=0)
	# Turn data into binary matrix: 0 is below, 1 is above the threshold
	binarized_data = dataframe.gt(thresholds).astype(int)
	return binarized_data


# Theshhold data with thresholding method without binarising
def thresholding(dataframe):
	# Compute threshold with thresholding method for each column
	thresholds = dataframe.apply(lambda x: threshold(x, "otsus"), axis=0)
	thresholded_data = dataframe.apply(lambda x: map(lambda y: 0 if y < thresholds.get(x.name) else y, x))
	return thresholded_data

def threshold(x, method):
	# Estimate number of bins according to Freedman Diaconis
	q_low, q_up = np.percentile(x, [25, 75])
	h = 2 * ((q_up - q_low)/len(x)**(1/3))
	fd_bins = math.ceil((max(x) - min(x)) / h)

	# Calculate list with thresholds
	if method == "otsus":
		return skif.threshold_otsu(x, nbins=fd_bins)
	elif method == "isodata":
		return skif.threshold_isodata(x, nbins=fd_bins)
	elif method == "yen":
		return skif.threshold_yen(x, nbins=fd_bins)
	elif method == "li":
		return skif.threshold_li(x)
	else:
		raise ValueError("Wrong Value for method in threshold()")


# Convert dataframe into NumPy array and transpose it. Samples: Images, Features: Pixel
def to_transposed_ndarray(dataframe):
	return dataframe.as_matrix().transpose()

# Convert dataframe into NumPy array. Samples: Pixel, Features: Images
def to_ndarray(dataframe):
	return dataframe.as_matrix()


# Needs: rows == pixel
# Pruning matrix area away could be automated with xml area identification
def prune_dataframe(dataframe, pruneRowTuple, pruneColTuple):
	# Convert None into a valid value for comparison with the effect that no borders are set
	row_min = pruneRowTuple[0] if pruneRowTuple[0] != None else -np.inf
	row_max = pruneRowTuple[1] if pruneRowTuple[1] != None else np.inf
	col_min = pruneColTuple[0] if pruneColTuple[0] != None else -np.inf
	col_max = pruneColTuple[1] if pruneColTuple[1] != None else np.inf

	grid_x = np.array(dataframe.index.get_level_values("grid_x"))
	grid_y = np.array(dataframe.index.get_level_values("grid_y"))

	# Iterate through all rows/pixel and check if they are in your pruning borders
	pruned_rows = [idx for idx, x in enumerate(zip(grid_x, grid_y)) if
				   row_min <= x[0] <= row_max
				   and col_min <= x[1] <= col_max]

	# Return a dataframe that consist of selected rows/pixelpositions/pixels
	return dataframe.iloc[pruned_rows]


def clear_matrix_area(dataframe, clear_span_x, clear_span_y):
	grid_x = dataframe.index.get_level_values("grid_x")
	grid_y = dataframe.index.get_level_values("grid_y")
	clear_area = [i for i,x in enumerate(zip(grid_x, grid_y))
				  if clear_span_x[0] <= x[0] <= clear_span_x[1]
				  and clear_span_y[0] <= x[1] <= clear_span_y[1]]
	dataframe.drop(dataframe.index[clear_area], inplace=True)


def clear_pixel_offset(dataframe):
	for lvl in dataframe.index.names:
		clear = dataframe.index.get_level_values(lvl) - dataframe.index.get_level_values(lvl).min()
		rename_dict = {dataframe.index.get_level_values(lvl)[i] : clear[i]
					   for i in range(clear.size)}
		dataframe.rename(index=rename_dict, level=lvl, inplace=True)


def denoiser(dataframe, method, thresholder = None, squared=False):
	grid_x = np.array(dataframe.index.get_level_values("grid_x"))
	x_max = grid_x.max()
	grid_y = np.array(dataframe.index.get_level_values("grid_y"))
	y_max = grid_y.max()
	img = np.zeros((y_max + 1, x_max + 1))

	if method == "threshold":
		# Square data to increase data range, first separation between low and high signal
		if squared:
			dataframe = dataframe ** 2
		for _, series in dataframe.T.iterrows():
			values = series.values
			# Determine number of bins
			q_low, q_up = np.percentile(values, [25, 75])
			h = 2 * ((q_up - q_low) / len(values) ** (1 / 3))
			fd_bins = math.ceil((max(values) - min(values)) / h)
			#n = len(values)
			#fd_bins = math.ceil(n ** (1 / 3))

			# Calculate intensity threshold
			# li -> isodata -> otsus (konservativ -> radikal)
			if thresholder == "otsus":
				t = skif.threshold_otsu(values, nbins=fd_bins)
			elif thresholder == "isodata":
				t = skif.threshold_isodata(values, nbins=fd_bins)
			elif thresholder == "li":
				t = skif.threshold_li(values)
			else:
				raise ValueError("Incorrect thresholder method in denoiser.")

			# Apply threshold
			series[series < t] = 0
		# Transform data back to its original range
		if squared:
			dataframe = np.sqrt(dataframe)
		return dataframe
	else:
		for img_name, intensities in dataframe.T.iterrows():
			# Create "digital" image
			img[(grid_y, grid_x)] = intensities

			# Use filter on each image
			if method == "chambolle":
				#TV-Chambolle denoising
				# eps=0.00002 bietet schaerfere Kanten als default von 0.0002
				img = res.denoise_tv_chambolle(img, weight=25)
			elif method == "bregman-iso":
				#TV-Bregman denoising
				img = res.denoise_tv_bregman(img, weight=0.05, isotropic=True)
			elif method == "bregman-aniso":
				#TV-Bregman anisotropic denoising
				img = res.denoise_tv_bregman(img, weight=0.05, isotropic=False)
			elif method == "bilateral":
				pass
				#Bilateral filter denoising
				#img = img.astype("float32")
				#img = cv2.bilateralFilter(img, 3, 60,60)
			elif method == "bilateral-adaptive":
				pass
				#img = cv2.adaptiveBilateralFilter(src=img, ksize=(3,3), sigmaSpace=50) #Needs a unsigned 8Bit image with 1 or 3 channels -> Greyscale or RGB
			elif method == "median":
				#Median filter
				img = filters.median_filter(img, size = (5,5))
			elif method == "gaussian-blur":
				#Gaussian Blur
				gauss_mask = np.array([[1,2,1],[2,4,2],[1,2,1]]) * 1/16
				img = filters.convolve(img,gauss_mask)
			else:
				raise ValueError("Wrong Value for method in denoiser()")

		# Exchange dataframe values with new values
		for idx in range(dataframe.shape[0]):
			pos_x = dataframe.index.get_level_values("grid_x")[idx]
			pos_y = dataframe.index.get_level_values("grid_y")[idx]
			dataframe[img_name][idx] = img[pos_y][pos_x]

		return dataframe