from __future__ import print_function, division

import itertools
import networkx as nx
import igraph as ig
from collections import defaultdict
from sklearn.cluster import k_means
import scipy.cluster.hierarchy as hac
import matplotlib.pyplot as plt
import numpy as np
import community as louvain

# Generate a graph based on adjacency matrix
def base_graph_structure(h5_data, adjacency_matrix):
	# Create the actual graph
	ig_G = ig.Graph.Adjacency(adjacency_matrix.astype(bool).tolist(), mode=ig.ADJ_UNDIRECTED)

	# Set name attribute for vertices based on m/z values in h5 data
	ig_G.vs["value"] = h5_data.columns

	# Set edge weights based on similarity measure in adjacency matrix
	for edge in ig_G.es:
		edge["weight"] = adjacency_matrix[edge.tuple[0]][edge.tuple[1]]

	print("")
	print("Base Graph:")
	print(ig.summary(ig_G))

	return ig_G


# Generate a list of edges from a list of nodes
def edgelist_generator(edgelist, nodelist, adjacency_matrix):
	nodelist = list(nodelist)
	for idx_x, vertex_x in enumerate(nodelist):
		for vertex_y in nodelist[idx_x+1:]:
			if adjacency_matrix[vertex_x][vertex_y] > 0:
				edgelist.append((vertex_x, vertex_y))
	return edgelist


# Calculate a new graph, where each community is reduced to one vertex
def community_graph(vertex_clustering_object, graph, edge_combination):
	c_graph = vertex_clustering_object.cluster_graph(combine_vertices="first", combine_edges=edge_combination)
	for c in vertex_clustering_object:
		# Needed because a combination of _in and list attribute causes an numpy any/all error
		c_graph.vs.find(value_in=graph.vs.select(c)["value"])["dummy"] = graph.vs.select(c)["value"]
	c_graph.vs["value"] = c_graph.vs["dummy"]
	del c_graph.vs["dummy"]
	## Rename membership to community
	#c_graph.vs["community"] = c_graph.vs["membership"]
	#del c_graph.vs["membership"]
	return c_graph


# Calculate Louvain method for community detection
# Level is calculated from behind, i.e. 0 is the highest level, i.e. level-n.
def calc_louvain(adj_matrix, level = 0, return_c_graph = False):
	nx_G = nx.from_numpy_matrix(adj_matrix)
	dendro = louvain.generate_dendrogram(nx_G, randomize=False) #Maybe set tandomize True

	if len(dendro) - level - 1 < 0:
		raise Exception("The given Level is too deep. The maximum is: " + str(len(dendro)-1))

	communities = louvain.partition_at_level(dendro, len(dendro) - level - 1)
	number_communities = max(communities, key = lambda x: communities[x]) + 1

	# Maybe unnecessary after some code rework and unification
	community_list = []
	for i in range(number_communities):
		grp_list = []
		for grp in communities:
			if communities[grp] == i:
				grp_list.append(grp)
		else:
			if grp_list:
				community_list.append(grp_list)

	community_level_G = louvain.induced_graph(communities, nx_G)

	if return_c_graph:
		c_level_graph = nx.adjacency_matrix(community_level_G)
	else:
		c_level_graph = None

	return community_list, c_level_graph

	#communities = louvain.best_partition(nx_G, resolution=None)


# Calculate Clique Percolation Method for community detection
def calc_cpm_communities(h5_data, adjacency_matrix, clique_size):
	print("Clique Size: " + str(clique_size))

	# Create the actual graph
	nx_G = nx.Graph(adjacency_matrix)

	# Clique Percolation Method (CPM) for community detection
	cpm_communities = list(nx.k_clique_communities(nx_G, clique_size, nx.find_cliques(nx_G)))

	# Create Graph based just on the found communities
	cpm_community_G = ig.Graph()
	# Create vertices
	cpm_community_G.add_vertices(range(len(adjacency_matrix)))
	# Set name attribute for vertices based on m/z values in h5 data
	cpm_community_G.vs["name"] = h5_data.columns
	# Create edges, assign memberships and find vertices with multiple memberships
	cpm_community_G.vs["membership"] = ""

	# Prepare empty edgelist
	edgelist = []
	for c in range(len(cpm_communities)):
		# Create edges, overlaps can create multiedges
		edgelist = edgelist_generator(edgelist, cpm_communities[c], adjacency_matrix)

		# Find vertices with multiple community membership and assign all memberships
		multi_members = []
		for v in cpm_communities[c]:
			v = int(v)
			if cpm_community_G.vs.select(v)["membership"][0] != "":
				multi_members.append(v)
				cpm_community_G.vs.select(v)["membership"] = cpm_community_G.vs.select(v).get_attribute_values("membership")[0] + "," + str(c)
			else:
				cpm_community_G.vs.select(v)["membership"] = str(c)

	# Delete multiedges with Set-Operator and set them into the Graph
	cpm_community_G.add_edges(set(edgelist))

	# Set edge weights based on similarity measure in adjacency matrix
	for edge in cpm_community_G.es:
		edge["weight"] = adjacency_matrix[edge.tuple[0]][edge.tuple[1]]

	# Set membership value for vertices without community
	without_community = list(set(cpm_community_G.vs.indices) - set(itertools.chain.from_iterable(cpm_communities)))
	for v in range(len(without_community)):
		cpm_community_G.vs.select(without_community[v])["membership"] = str(len(cpm_communities) + v)

	# Create membership list
	membership_list = []
	overlap_node_counter = 0
	for v in cpm_community_G.vs:
		membership = v["membership"].split(",")
		if len(membership) > 1:
			overlap_node_counter += 1
			#TODO: A change is needed, because putting all overlap vertices in one community affects the modularity in an unknown way!
			#TODO: Insert a different membership behaviour for vertices with multiple memberships HERE!
			membership_list.append(len(cpm_communities) + len(without_community))
		else:
			membership_list.append(int(membership[0]))

	print("")
	print("Community Graph:")
	print(ig.summary(cpm_community_G))

	# Print number of communities
	print("")
	print("Number of Communities: " + str(len(cpm_communities)))

	print("")
	print("Number of Nodes without Communities: " + str(len(without_community)))

	print("")
	print("Number of Overlapping Nodes: " + str(overlap_node_counter))

	# Code to maintain all edges, without all edges the modularity is influenced
	base_graph = base_graph_structure(h5_data, adjacency_matrix)
	cpm_communities_vc = ig.VertexClustering(base_graph, membership=membership_list)

	# Calculate unweighted modularity
	modularity = cpm_communities_vc.modularity
	# Calculate weighted modularity
	#modularity = base_graph.modularity(cpm_communities_vc, weights=base_graph.es["weight"])

	print("")
	print("Modularity: " + str(modularity))

	return cpm_community_G, cpm_communities_vc


# Calculate Modularity Matrix based Method for community detection
def calc_mmm_communities(h5_data, adjacency_matrix, cluster_boundary):
	# Create the actual graph
	mmm_community_G = base_graph_structure(h5_data, adjacency_matrix)

	# Modularity Matrix based Method (MMM) for community detection
	mmm_communities = mmm_community_G.community_leading_eigenvector(clusters=cluster_boundary, weights=mmm_community_G.es["weight"]) # Just weights = "weight" works as well

	print("")
	print("Community Graph:")
	print(ig.summary(mmm_community_G))
	print("")
	print("Number of Communities: " + str(len(list(mmm_communities))))

	#Add community membership as attribute
	for v in mmm_community_G.vs:
		v["membership"] = mmm_communities.membership[v.index]

	# Calculate unweighted modularity
	modularity = mmm_communities.modularity
	# Calculate weighted modularity
	#modularity =  mmm_community_G.modularity(mmm_communities, weights=mmm_community_G.es["weight"])
	print("")
	print("Modularity: " + str(modularity))

	return mmm_community_G, mmm_communities


# Calculate Hierarchical Clustering for community detection
def calc_hac_communities(h5_data, adjacency_matrix, linkage_method = "average", metric = "correlation" , plot_flag = True, threshold = None):

	distance_matrix = 1 - adjacency_matrix
	# Create condensed distance matrix
	# A condensed distance matrix is a flat array containing the upper triangular of the distance matrix. (SciPy)
	distance_array = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
	# Alternative to the upper version
	#np.fill_diagonal(distance_matrix, 0.0)
	#distance_matrix = np.around(distance_matrix, 7) #Attention! Round affects clustering
	#distance_array = squareform(distance_matrix)

	# Linkage can be single, complete, average, weighted
	# Calculate linkage matrix
	z = hac.linkage(distance_array, linkage_method, metric)

	# Creation of the actual graph
	hac_community_G = base_graph_structure(h5_data, adjacency_matrix)

	# Calculate dendrogram-cut based on modularity optimization
	threshold_list = []
	for x in range(1,len(adjacency_matrix)+1):
		memberships = hac.fcluster(z, x, criterion="maxclust")
		threshold_list.append(modularity_trsh(memberships, hac_community_G))

	if plot_flag == True:
		plt.figure()
		plt.xticks(range(0, len(adjacency_matrix)), range(1, len(adjacency_matrix)+1))
		plt.title("modularity")
		plt.plot(threshold_list)
		plt.figure()
		hac.dendrogram(z)
		plt.show()

	if threshold == None:
		print("")
		print("Threshold by Modularity used!")
		# +1 because modularity calculation starts with 1 cluster instead of 0, but indexing starts with 0
		threshold = threshold_list.index(max(threshold_list))+1
	else:
		print("")
		print("Threshold set manually!")

	# Calculate Hierarchical Clustering
	#membership_list = hac.fclusterdata(data_matrix, threshold, criterion="maxclust", metric=metric, method=linkage_method)
	membership_list = hac.fcluster(z,threshold,criterion="maxclust")

	# Reduce each membership value by one
	# fcluster starts with membership number one, for transformation into ig.VertexClustering a starting membership of zero is needed
	membership_list = map(lambda x: x - 1, membership_list)

	hac_communities = ig.VertexClustering(hac_community_G, membership=membership_list)

	print("")
	print("Community Graph:")
	print(ig.summary(hac_community_G))

	print("")
	print("Threshold of Dendrogramm Cut: " + str(threshold))

	# Add community membership as attribute
	for vertex in hac_community_G.vs:
		vertex["membership"] = hac_communities.membership[vertex.index]

	print("")
	print("Number of Communities: " + str(len(list(hac_communities))))

	# Calculate unweighted modularity
	modularity = hac_communities.modularity
	# Calculate weighted modularity
	# modularity = hac_community_G.modularity(hac_communities, weights=hac_community_G.es["weight"])
	print("")
	print("Modularity: " + str(modularity))

	return hac_community_G, hac_communities


# Calculate k-means
def kmeans_clustering(h5_data, data_matrix, adjacency_matrix, k):
	centroids, membership_list, inertia = k_means(data_matrix, n_clusters=k)

	# Defaultdict with cluster-membership as key and a list with all members as value
	kmeans_communities = defaultdict(list)
	for x in range(len(membership_list)):
		#mass_float = round(float(data.columns[x]),3)
		#kmeans_communities[membership_list[x]].append(mass_float)
		kmeans_communities[membership_list[x]].append(x)

	# Creation of the actual graph
	kmeans_community_G = base_graph_structure(h5_data, adjacency_matrix)
	# Set communities based on the cluster found by k-means
	kmeans_communities = ig.VertexClustering(kmeans_community_G, membership=membership_list)

	return kmeans_community_G, kmeans_communities, membership_list, centroids


def modularity_trsh(memberships, graph):
	hac_communities = ig.VertexClustering(graph, membership=memberships)
	# Return unweighted modularity
	return hac_communities.modularity
	# Return weighted modularity
	#return graph.modularity(hac_communities, weights=graph.es["weight"])


def igraph_lou(graph, level = 0):
	print(graph.community_multilevel(return_levels=True))
	return graph.community_multilevel(return_levels=True)[level]


def calc_infomap(G):
	return G.community_infomap()