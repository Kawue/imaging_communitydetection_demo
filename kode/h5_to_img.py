from __future__ import print_function, division
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import cnames


def imgWriter(h5_data, save_path, scaling="all", communities=None, colorscale_boundary=(0, 100),
			  prune_row_tuple=(0, None), prune_col_tuple=(0, None)):
	if not os.path.isdir(save_path):
		os.makedirs(save_path)

	if scaling not in ["all", "group", "single"]:
		raise Exception("Scaling has to be \"all\", \"group\" or \"single\"")

	# if prune_row_tuple == None:
	#    prune_row_tuple = (0,None)
	# if prune_col_tuple == None:
	#    prune_col_tuple = (0,None)
	rowStart, rowEnd = prune_row_tuple
	colStart, colEnd = prune_col_tuple

	# if colorscale_boundary == None:
	#    colorscale_boundary = (0,100)

	# http://matplotlib.org/examples/color/colormaps_reference.html
	color_map = plt.cm.ScalarMappable(plt.Normalize(), cmap=matplotlib.cm.Greys_r)  # Greys_r, viridis

	if scaling == "all":
		# Set limits for color mapping, limit is defined by the set of all images
		color_map.set_clim(np.percentile(h5_data, colorscale_boundary))

	grid_x = np.array(h5_data.index.get_level_values("grid_x"))
	x_max = grid_x.max()
	grid_y = np.array(h5_data.index.get_level_values("grid_y"))
	y_max = grid_y.max()

	img = np.zeros((y_max + 1, x_max + 1, 4))

	if scaling == "group":
		if not communities:
			raise Exception("No Communities given, scaling by group is not possible!")

		for com in communities:
			color_map.set_clim(np.percentile(h5_data.iloc[:, com], colorscale_boundary))
			for img_name, intensities in h5_data.iloc[:, com].T.iterrows():
				# map component to color scale, the bytes flag enabels mapping to 0..255 in uint8 instead of 0..1 in float
				# Convert #pixel intensities to #pixel 4-tuple
				intensities = color_map.to_rgba(intensities)

				# for each 4-tuple ( = #pixel) map it into the corresponding (y,x) - pixel
				# Structure of img -> grid_y times 2D-arrays of size grid_x times 4 (4 = rgba)
				# -> Each row of array grid_x is one (x,y) - Pixel, consisting of 4 values for r,g,b,a
				img[(grid_y, grid_x)] = intensities

				plt.imsave(os.path.join(save_path, str(img_name) + '.png'), img[rowStart:rowEnd, colStart:colEnd, :])
	else:
		# data.T -> Rows = m/z Images; Columns = Pixelintensities
		# iterrows() -> Tuple: (index, series)
		# After Transpose -> index = m/z value; series = pixelintensities
		for img_name, intensities in h5_data.T.iterrows():
			if scaling == "single":
				# Set limits for color mapping, limit is defined for every image
				color_map.set_clim(np.percentile(intensities, colorscale_boundary))

			# map component to color scale, the bytes flag enabels mapping to 0..255 in uint8 instead of 0..1 in float
			# Convert #pixel intensities to #pixel 4-tuple
			intensities = color_map.to_rgba(intensities)

			# for each 4-tuple ( = #pixel) map it into the corresponding (y,x) - pixel
			# Structure of img -> grid_y times 2D-arrays of size grid_x times 4 (4 = rgba)
			# -> Each row of array grid_x is one (x,y) - Pixel, consisting of 4 values for r,g,b,a
			img[(grid_y, grid_x)] = intensities

			plt.imsave(os.path.join(save_path, str(img_name) + '.png'), img[rowStart:rowEnd, colStart:colEnd, :])


def rgb_img_writer(red_img, green_img, blue_img, colorscale_boundary, prune_row_tuple, prune_col_tuple, h5_data,
				   save_path):
	if not os.path.isdir(save_path):
		os.makedirs(save_path)

	if prune_row_tuple == None:
		prune_row_tuple = (0, None)
	if prune_col_tuple == None:
		prune_col_tuple = (0, None)
	rowStart, rowEnd = prune_row_tuple
	colStart, colEnd = prune_col_tuple

	if colorscale_boundary == None:
		colorscale_boundary = (0, 100)

	grid_x = np.array(h5_data.index.get_level_values("grid_x"))
	x_max = grid_x.max()
	grid_y = np.array(h5_data.index.get_level_values("grid_y"))
	y_max = grid_y.max()

	color_scheme = plt.cm.Greys_r
	color_map = plt.cm.ScalarMappable(norm=plt.Normalize(), cmap=color_scheme)
	color_map.set_clim(np.percentile(red_img, colorscale_boundary))
	r = color_map.to_rgba(red_img)
	color_map.set_clim(np.percentile(green_img, colorscale_boundary))
	g = color_map.to_rgba(green_img)
	color_map.set_clim(np.percentile(blue_img, colorscale_boundary))
	b = color_map.to_rgba(blue_img)

	rgb = np.zeros((y_max + 1, x_max + 1, 3))
	rr = np.zeros((y_max + 1, x_max + 1, 3))
	gg = np.zeros((y_max + 1, x_max + 1, 3))
	bb = np.zeros((y_max + 1, x_max + 1, 3))

	rgb[(grid_y, grid_x, 0)] = r[:, 0]
	rgb[(grid_y, grid_x, 1)] = g[:, 0]
	rgb[(grid_y, grid_x, 2)] = b[:, 0]

	rr[(grid_y, grid_x, 0)] = r[:, 0]
	rr[(grid_y, grid_x, 1)] = r[:, 1]
	rr[(grid_y, grid_x, 2)] = r[:, 2]

	gg[(grid_y, grid_x, 0)] = g[:, 0]
	gg[(grid_y, grid_x, 1)] = g[:, 1]
	gg[(grid_y, grid_x, 2)] = g[:, 2]

	bb[(grid_y, grid_x, 0)] = b[:, 0]
	bb[(grid_y, grid_x, 1)] = b[:, 1]
	bb[(grid_y, grid_x, 2)] = b[:, 2]

	plt.imsave(os.path.join(save_path, "r.png"), rr[rowStart:rowEnd, colStart:colEnd, :])
	plt.imsave(os.path.join(save_path, "g.png"), gg[rowStart:rowEnd, colStart:colEnd, :])
	plt.imsave(os.path.join(save_path, "b.png"), bb[rowStart:rowEnd, colStart:colEnd, :])
	plt.imsave(os.path.join(save_path, "rgb.png"), rgb[rowStart:rowEnd, colStart:colEnd, :])


def pca_img_writer(pcr_data, n_components, colorscale_boundary, prune_row_tuple, prune_col_tuple, h5_data, save_path):
	if not os.path.isdir(save_path):
		os.makedirs(save_path)

	if prune_row_tuple == None:
		prune_row_tuple = (0, None)
	if prune_col_tuple == None:
		prune_col_tuple = (0, None)
	rowStart, rowEnd = prune_row_tuple
	colStart, colEnd = prune_col_tuple

	if colorscale_boundary == None:
		colorscale_boundary = (0, 100)

	color_scheme = plt.cm.Greys_r
	color_map = plt.cm.ScalarMappable(norm=plt.Normalize(), cmap=color_scheme)

	grid_x = np.array(h5_data.index.get_level_values("grid_x"))
	x_max = grid_x.max()
	grid_y = np.array(h5_data.index.get_level_values("grid_y"))
	y_max = grid_y.max()

	img = np.zeros((y_max + 1, x_max + 1, 4))
	# Set limits for color mapping
	# color_map.set_clim(np.percentile(pcr_data[:,0:n_components], plot_boundary))

	for idx, intensities in enumerate(pcr_data[:, 0:n_components].T, start=1):
		# Set individual limits for color mapping for each component
		color_map.set_clim(np.percentile(intensities, colorscale_boundary))
		intensities = color_map.to_rgba(intensities)
		# Works if PCR data values have the same order as the h5_data dataframe
		img[(grid_y, grid_x)] = intensities

		plt.imsave(os.path.join(save_path, "component_" + str(idx) + ".png"),
						   img[rowStart:rowEnd, colStart:colEnd, :])


def write_mean_image(communities, colorscale_boundary, prune_row_tuple, prune_col_tuple, h5_data, save_path):
	if not os.path.isdir(save_path):
		os.makedirs(save_path)

	if prune_row_tuple == None:
		prune_row_tuple = (0, None)
	if prune_col_tuple == None:
		prune_col_tuple = (0, None)
	rowStart, rowEnd = prune_row_tuple
	colStart, colEnd = prune_col_tuple

	if colorscale_boundary == None:
		colorscale_boundary = (0, 100)

	grid_x = np.array(h5_data.index.get_level_values("grid_x"))
	x_max = grid_x.max()
	grid_y = np.array(h5_data.index.get_level_values("grid_y"))
	y_max = grid_y.max()

	img = np.zeros((y_max + 1, x_max + 1, 4))

	color_scheme = plt.cm.Greys_r
	color_map = plt.cm.ScalarMappable(norm=plt.Normalize(), cmap=color_scheme)
	# Scaling over all data values vs scaling over all community members(see below)
	color_map.set_clim(np.percentile(h5_data.as_matrix(), colorscale_boundary))

	for membership in range(len(communities)):
		idx_list = [i for i, x in enumerate(communities.membership) if x == membership]
		community_frame = h5_data.iloc[:, idx_list]
		community_mean = np.mean(community_frame, axis=1)

		color_map.set_clim(np.percentile(community_mean, colorscale_boundary))

		intensities = color_map.to_rgba(community_mean)

		img[(grid_y, grid_x)] = intensities
		plt.imsave(os.path.join(save_path, "mean_community_" + str(membership) + ".png"),
						   img[rowStart:rowEnd, colStart:colEnd, :])


def write_max_image(communities, colorscale_boundary, prune_row_tuple, prune_col_tuple, h5_data, save_path):
	if not os.path.isdir(save_path):
		os.makedirs(save_path)

	if prune_row_tuple == None:
		prune_row_tuple = (0, None)
	if prune_col_tuple == None:
		prune_col_tuple = (0, None)
	rowStart, rowEnd = prune_row_tuple
	colStart, colEnd = prune_col_tuple

	if colorscale_boundary == None:
		colorscale_boundary = (0, 100)

	grid_x = np.array(h5_data.index.get_level_values("grid_x"))
	x_max = grid_x.max()
	grid_y = np.array(h5_data.index.get_level_values("grid_y"))
	y_max = grid_y.max()

	img = np.zeros((y_max + 1, x_max + 1, 4))

	color_scheme = plt.cm.viridis
	color_map = plt.cm.ScalarMappable(norm=plt.Normalize(), cmap=color_scheme)
	# Scaling over all data values vs scaling over all community members(see below)
	# color_map.set_clim(np.percentile(h5_data.as_matrix(), colorscale_boundary))

	for membership in range(len(communities)):
		idx_list = [i for i, x in enumerate(communities.membership) if x == membership]
		community_frame = h5_data.iloc[:, idx_list]
		community_max = np.max(community_frame, axis=1)

		color_map.set_clim(np.percentile(community_max, colorscale_boundary))

		intensities = color_map.to_rgba(community_max)

		img[(grid_y, grid_x)] = intensities
		plt.imsave(os.path.join(save_path, "max_community_" + str(membership) + ".png"),
						   img[rowStart:rowEnd, colStart:colEnd, :])


# NEEDS TO BE CHECKED AGAIN, PRETTY OLD
# implementieren, dass nur gruppen > 1 eine farbe bekommen, alles andere in einheitsfarbe, e.g. grau
def imgSegmentationWriter(graph, communities, mainPath, h5_data):
	if not os.path.isdir(os.path.join(os.path.dirname(mainPath), "segmentation-image")):
		os.makedirs(os.path.join(os.path.dirname(mainPath), "segmentation-image"))
	print(h5_data.index)
	# Set color scheme and color map
	color_scheme = plt.cm.viridis
	color_map = plt.cm.ScalarMappable(norm=plt.Normalize(), cmap=color_scheme)

	# Get pixel positions to map each element to its respective position
	grid_x = np.array(h5_data.index.get_level_values("grid_x"))
	x_max = grid_x.max()
	grid_y = np.array(h5_data.index.get_level_values("grid_y"))
	y_max = grid_y.max()

	img = np.zeros((y_max + 1, x_max + 1, 4))

	save_path = "segmentation-image"

	# Count occurence of values and create a True/False array for bigger than one. Sum counts True as one and False as zero.
	# Minus one is required because membership starts at zero.
	nb_colorlimit = (np.bincount(communities.membership) > 1).sum() - 1
	print("number comms: " + str(nb_colorlimit))
	# Membership always starts with 0 and ends with number of communities > 2
	color_map.set_clim(0, nb_colorlimit)

	# Scale each pixel according to its community membership (order of community.membership corresponds to order in dataframe)
	rgba_pixel = color_map.to_rgba(communities.membership)
	# Set a color, like light gray, for communities with just one member, since they most likely do not code for special structures
	brush_out_color = mcolors.to_rgba_array(cnames["lightgray"])[0]
	for idx, memb in enumerate(communities.membership):
		if memb > nb_colorlimit:
			rgba_pixel[idx] = brush_out_color

	img[(grid_y, grid_x)] = rgba_pixel

	plt.imsave(os.path.join(save_path, "segmentationMap.png"), img)


def imgSegmentationWriter_graphless(communities, membership_list, mainPath, h5_data):
	if not os.path.isdir(os.path.join(os.path.dirname(mainPath), "segmentation-image")):
		os.makedirs(os.path.join(os.path.dirname(mainPath), "segmentation-image"))
	print(h5_data.index)
	# Set color scheme and color map
	color_scheme = plt.cm.viridis
	color_map = plt.cm.ScalarMappable(norm=plt.Normalize(), cmap=color_scheme)

	# Get pixel positions to map each element to its respective position
	grid_x = np.array(h5_data.index.get_level_values("grid_x"))
	x_max = grid_x.max()
	grid_y = np.array(h5_data.index.get_level_values("grid_y"))
	y_max = grid_y.max()

	img = np.zeros((y_max + 1, x_max + 1, 4))

	save_path = "segmentation-image"
	print(communities.keys())
	# Count occurence of values and create a True/False array for bigger than one. Sum counts True as one and False as zero.
	# Minus one is required because membership starts at zero.
	nb_colorlimit = np.max(list(communities.keys()))
	print("number comms: " + str(nb_colorlimit))
	# Membership always starts with 0 and ends with number of communities > 2
	color_map.set_clim(0, nb_colorlimit)
	print(membership_list)
	# Scale each pixel according to its community membership (order of community.membership corresponds to order in dataframe)
	rgba_pixel = color_map.to_rgba(membership_list)
	# Set a color, like light gray, for communities with just one member, since they most likely do not code for special structures
	brush_out_color = mcolors.to_rgba_array(cnames["lightgray"])[0]
	for idx, memb in enumerate(membership_list):
		if memb > nb_colorlimit:
			rgba_pixel[idx] = brush_out_color

	img[(grid_y, grid_x)] = rgba_pixel

	plt.imsave(os.path.join(save_path, "segmentationMap.png"), img)