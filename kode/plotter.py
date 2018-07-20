from __future__ import print_function, division

import os
import random
import igraph as ig

def plot_graph(graph, communities, save_plot, name_plot, save_gml, name_gml, show_plot, save_path):
	# Set a random seed to get the same plot multiple times
	random.seed(123)

	# Rescale weights for use as edge lengths
	weight_scale_length = ig.rescale(graph.es["weight"], out_range=(0, 1))
	# Set layout
	layout = graph.layout_fruchterman_reingold(weights=weight_scale_length)
	# Use weights for edge thickness
	weight_scale_thickness = graph.es["weight"]

	# Plot graph structure and color it according to its communities
	if save_plot:
		if not os.path.isdir(save_path):
			os.makedirs(save_path)
		# Create and save plot
		plot = ig.plot(communities,
		               vertex_label=graph.vs["value"],
		               layout=layout,
		               edge_width=weight_scale_thickness,
		               target=os.path.join(save_path, name_plot + ".png"))
		if show_plot:
			plot.show()
	else:
		if show_plot:
			# Create and show plot
			ig.plot(communities,
		               vertex_label=graph.vs["value"],
		               layout=layout,
		               edge_width=weight_scale_thickness)

	if save_gml:
		if not os.path.isdir(save_path):
			os.makedirs(save_path)
		graph.write_graphml(os.path.join(save_path, name_gml + ".graphml"))
