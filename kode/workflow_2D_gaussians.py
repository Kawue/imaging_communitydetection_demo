from scipy.stats.mstats import winsorize
from kode.preprocessing import *
from kode.similarity_measures import *
from kode.edge_reduction import *
from kode.community_detection import *
from kode.plotter import *
from kode.json_factory import *
from kode.mmm_own import *
from kode.miscellaneous import *

def workflow_2D_gaussians():
	# Create gaussian images
	h5_data = gaussian_2d_images("circles")

	# Winsorize data
	winsorize(h5_data, limits=(0, 0.01), axis=0, inplace=True)

	# Convert hdf data in numpy array; desired structure Samples = m/z-images, Features = Pixel
	data = to_transposed_ndarray(h5_data.copy())

	# Calculate similarity matrix
	similarity_matrix = calc_pearson_correlation(data)
	print("Similarity Matrix Calculation Done!")

	# Transform similarity matrix to adjacency matrix
	adjacency_matrix, edge_reduction_threshold = transform_by_pca(similarity_matrix, [-1, 1], 200)
	print("Adjecency Matrix Calculation Done!")

	# Transform weighted adjacency matrix to unweighted
	adjacency_matrix = adjacency_matrix.astype(bool).astype(float)

	# Calculate communities
	communitiy_list = leading_eigenvector_community(adjacency_matrix, None, False, False, None)

	# Sort communities by length
	sorted_community_list = sorted(communitiy_list, key=len)

	# Calculate membership list
	membership_list = []
	for vertex in range(len(adjacency_matrix)):
		for membership_id, community in enumerate(sorted_community_list):
			if vertex in community:
				membership_list.append(membership_id)

	# Calculate graph
	graph = base_graph_structure(h5_data, adjacency_matrix)

	# Calculate communities as clustering object
	communities = ig.VertexClustering(graph, membership_list)

	print("")
	print("Community Graph:")
	print(ig.summary(graph))
	print("")
	print("Number of Communities: " + str(len(list(communities))))

	# Add community membership as attribute
	for v in graph.vs:
		v["membership"] = communities.membership[v.index]

	# Calculate unweighted modularity
	modularity = communities.modularity
	print("")
	print("Modularity: " + str(modularity))
	print("Community Calculation Done!")

	# Calculate community-graph
	c_graph = community_graph(communities, graph, "mean")

	# Plot
	save_path = "../results/"
	file_name = "2D_gaussians_" + str(edge_reduction_threshold)
	plot_graph(graph, communities,
			   False, file_name,
			   False, file_name,
			   True, os.path.join(os.path.dirname(os.path.realpath(__file__)), save_path))

	# Create JSON
	build_json_grine(graph, communities, c_graph, h5_data, file_name,
			   os.path.join(os.path.dirname(os.path.realpath(__file__)), save_path))