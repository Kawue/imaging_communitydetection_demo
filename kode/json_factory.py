import simplejson as json
from collections import OrderedDict
import igraph as ig
import numpy as np
import os

def build_json_grine(graph, communities, c_graph, h5_data, name_json, save_path):
	json_dict = OrderedDict()

	mz_list = []
	for n in graph.vs:
		mz_list.append(str(round(n["value"], 3)))
	json_dict["mz_values"] = mz_list

	nodes_list = []
	for idx, ids in enumerate(communities):
		community_dict = OrderedDict()
		community_list = []
		for i, n in enumerate(graph.vs.select(ids)):
			n_dict = OrderedDict()
			n_dict["id"] = str(ids[i])
			n_dict["name"] = str(round(n["value"],3))
			n_dict["membership"] = "c" + str(idx)
			n_dict["mean_intensity"] = np.mean(h5_data[n["value"]])
			n_dict["degree"] = graph.degree(n)
			community_list.append(n_dict)
		community_dict["id"] = "c" + str(idx)
		community_dict["nodes"] = community_list
		nodes_list.append(community_dict)
	json_dict["c_nodes"] = nodes_list

	edges_list = []
	for e in graph.es:
		e_dict = OrderedDict()
		e_dict["source"] = str(e.tuple[0])
		e_dict["target"] = str(e.tuple[1])
		e_dict["weight"] = round(e["weight"],5)
		edges_list.append(e_dict)
	json_dict["edges"] = edges_list

	c_edges_list = []
	for e in c_graph.es:
		e_dict = OrderedDict()
		e_dict["source"] = "c" + str(e.tuple[0])
		e_dict["target"] = "c" + str(e.tuple[1])
		#e_dict["weight"]
		c_edges_list.append(e_dict)
	json_dict["c_edges"] = c_edges_list

	if not os.path.isdir(save_path):
		os.makedirs(save_path)

	path = os.path.join(save_path, name_json + ".json")

	f = open(path, "w")
	with f as outputfile:
		json.dump(json_dict, outputfile)



def build_json_palia(msi_frame_1, graph_1, community_list_1, c_graph_1,
					 msi_frame_2, graph_2, community_list_2, c_graph_2,
					 fips, save_path, name_json):
	nodes_g1, edges_g1, nodes_g1_only, edges_g1_only, \
	nodes_g2, edges_g2, nodes_g2_only, edges_g2_only, \
	nodes_common, edges_common = generate_node_edge_lists(msi_frame_1, msi_frame_2, graph_1, graph_2)
	json_dict = dict()

	# Evtl. statt Liste ein Dict benutzen und ID als Key.
	# Evtl. statt named_nodes und named_edges die id's statt namen verwenden.
	json_dict["graphs"] = \
		[
			{"id": "g1",
			 "nodes": nodes_g1,
			 "edges": edges_g1
			 },
			{"id": "g2",
			 "nodes": nodes_g2,
			 "edges": edges_g2}
		]

	nodes = []
	for i, n in enumerate(nodes_g1_only):
		d = dict()
		d["id"] = "n" + str(i) + "g1"
		d["value"] = n
		d["partners"] = [node["value"] for node in graph_1.vs.find(value_eq=n).neighbors()]
		d["degree"] = graph_1.vs.find(value_eq=n).degree()
		d["graph"] = ["g1"]
		d["community"] = graph_1.vs.find(value_eq=n)["membership"]
		d["shared"] = False
		nodes.append(d)

	for i, n in enumerate(nodes_g2_only):
		d = dict()
		d["id"] = "n" + str(i) + "g2"
		d["value"] = n
		d["partners"] = [node["value"] for node in graph_2.vs.find(value_eq=n).neighbors()]
		d["degree"] = graph_2.vs.find(value_eq=n).degree()
		d["graph"] = ["g2"]
		d["community"] = graph_2.vs.find(value_eq=n)["membership"]
		d["shared"] = False
		nodes.append(d)

	for i, n in enumerate(nodes_common):
		d = dict()
		d["id"] = "n" + str(i) + "g1g2"
		d["value"] = n
		d["partners"] = {
							"g1": [node["value"] for node in graph_1.vs.find(value_eq=n).neighbors()],
							"g2": [node["value"] for node in graph_2.vs.find(value_eq=n).neighbors()]
						}
		d["degree"] = {
						"g1": graph_1.vs.find(value_eq=n).degree(),
						"g2": graph_2.vs.find(value_eq=n).degree()
					}
		d["graph"] = ["g1", "g2"]
		d["community"] = {
							"g1": graph_1.vs.find(value_eq=n)["membership"],
							"g2": graph_2.vs.find(value_eq=n)["membership"]
						}
		d["shared"] = True
		nodes.append(d)

	json_dict["nodes"] = nodes

	edges = []
	for i, e in enumerate(edges_g1_only):
		d = dict()
		d["id"] = "e" + str(i) + "g1"
		d["source"] = [node["id"] for node in nodes if node["value"] == min(e)][0]
		d["target"] = [node["id"] for node in nodes if node["value"] == max(e)][0]
		d["weight"] = graph_1.es.find(_between=(graph_1.vs.select(value_eq=min(e)), graph_1.vs.select(value_eq=max(e))))["weight"]
		d["graph"] = ["g1"]
		d["scope"] = "within" if graph_1.vs.find(value_eq=e[0])["membership"] == graph_1.vs.find(value_eq=e[1])["membership"] else "between"
		d["shared"] = False
		if len([node["id"] for node in nodes if node["value"] == min(e)]) > 1 or len([node["id"] for node in nodes if node["value"] == max(e)]) > 1:
			print([node["id"] for node in nodes if node["value"] == min(e)])
			print([node["id"] for node in nodes if node["value"] == max(e)])
			print(len([node["id"] for node in nodes if node["value"] == min(e)]))
			print(len([node["id"] for node in nodes if node["value"] == max(e)]))
			raise ValueError
		edges.append(d)

	for i, e in enumerate(edges_g2_only):
		d = dict()
		d["id"] = "e" + str(i) + "g2"
		d["source"] = [node["id"] for node in nodes if node["value"] == min(e)][0]
		d["target"] = [node["id"] for node in nodes if node["value"] == max(e)][0]
		d["weight"] = graph_2.es.find(_between=(graph_2.vs.select(value_eq=min(e)), graph_2.vs.select(value_eq=max(e))))["weight"]
		d["graph"] = ["g2"]
		d["scope"] = "within" if graph_2.vs.find(value_eq=e[0])["membership"] == graph_2.vs.find(value_eq=e[1])["membership"] else "between"
		d["shared"] = False
		if len([node["id"] for node in nodes if node["value"] == min(e)]) > 1 or len([node["id"] for node in nodes if node["value"] == max(e)]) > 1:
			print([node["id"] for node in nodes if node["value"] == min(e)])
			print([node["id"] for node in nodes if node["value"] == max(e)])
			raise ValueError
		edges.append(d)

	for i, e in enumerate(edges_common):
		d = dict()
		d["id"] = "e" + str(i) + "g1g2"
		d["source"] = [node["id"] for node in nodes if node["value"] == min(e)][0]
		d["target"] = [node["id"] for node in nodes if node["value"] == max(e)][0]
		d["weight"] = {
						"g1": graph_1.es.find(_between=(graph_1.vs.select(value_eq=min(e)), graph_1.vs.select(value_eq=max(e))))["weight"],
						"g2": graph_2.es.find(_between=(graph_2.vs.select(value_eq=min(e)), graph_2.vs.select(value_eq=max(e))))["weight"]
					}
		d["graph"] = ["g1", "g2"]
		d["scope"] = {
						"g1": "within" if graph_1.vs.find(value_eq=e[0])["membership"] == graph_1.vs.find(value_eq=e[1])["membership"] else "between",
						"g2": "within" if graph_2.vs.find(value_eq=e[0])["membership"] == graph_2.vs.find(value_eq=e[1])["membership"] else "between"
					}
		d["shared"] = True
		if len([node["id"] for node in nodes if node["value"] == min(e)]) > 1 or len([node["id"] for node in nodes if node["value"] == max(e)]) > 1:
			print([node["id"] for node in nodes if node["value"] == min(e)])
			print([node["id"] for node in nodes if node["value"] == max(e)])
			raise ValueError
		edges.append(d)

	json_dict["edges"] = edges

	communities = []
	for i, c in enumerate(community_list_1):
		d = dict()
		d["id"] = "c" + str(c_graph_1.vs.find(value_eq=c)["membership"]) + "g1"
		d["graph"] = ["g1"]
		d["nodes"] = c
		d["partners"] = ["c" + str(node["membership"]) + "g1" for node in c_graph_1.vs.find(value_eq=c).neighbors()]
		communities.append(d)

	for i, c in enumerate(community_list_2):
		d = dict()
		d["id"] = "c" + str(c_graph_2.vs.find(value_eq=c)["membership"]) + "g2"
		d["graph"] = ["g2"]
		d["nodes"] = c
		d["partners"] = ["c" + str(node["membership"]) + "g2" for node in c_graph_2.vs.find(value_eq=c).neighbors()]
		communities.append(d)

	json_dict["communities"] = communities

	community_edges = []
	for i, e in enumerate(c_graph_1.es):
		d = dict()
		d["id"] = "ce" + str(i) + "g1"
		d["source"] = "c" + str(c_graph_1.vs[min(e.tuple)]["membership"]) + "g1"
		d["target"] = "c" + str(c_graph_1.vs[max(e.tuple)]["membership"]) + "g1"
		d["weight"] = e["weight"]
		d["graph"] = ["g1"]
		community_edges.append(d)

	for i, e in enumerate(c_graph_2.es):
		d = dict()
		d["id"] = "ce" + str(i) + "g2"
		d["source"] = "c" + str(min(e.tuple)) + "g2"
		d["target"] = "c" + str(max(e.tuple)) + "g2"
		d["weight"] = e["weight"]
		d["graph"] = ["g2"]
		community_edges.append(d)

	json_dict["communityedges"] = community_edges

	if not os.path.isdir(save_path):
		os.makedirs(save_path)
	path = os.path.join(save_path, name_json + ".json")
	f = open(path, "w")
	with f as outputfile:
		json.dump(json_dict, outputfile)
		

def generate_node_edge_lists(msi_frame_1, msi_frame_2, graph_1, graph_2):
	edges_g1 = [(msi_frame_1.columns[e.tuple[0]], msi_frame_1.columns[e.tuple[1]]) for e in graph_1.es]
	edges_g2 = [(msi_frame_2.columns[e.tuple[0]], msi_frame_2.columns[e.tuple[1]]) for e in graph_2.es]
	nodes_g1 = [msi_frame_1.columns[v.index] for v in graph_1.vs]
	nodes_g2 = [msi_frame_2.columns[v.index] for v in graph_2.vs]
	nodes_g1_only = set(nodes_g1) - set(nodes_g2)
	nodes_g2_only = set(nodes_g2) - set(nodes_g1)
	nodes_common = set(nodes_g1) & set(nodes_g2)
	edges_g1_only = set(edges_g1) - set(edges_g2)
	edges_g2_only = set(edges_g2) - set(edges_g1)
	edges_common = set(edges_g1) & set(edges_g2)
	print(edges_g1)
	print(edges_g2)
	print(nodes_g1)
	print(nodes_g2)
	print("len symdiff edges")
	print(len(set(edges_g1) ^ set(edges_g2)))
	print("len symdiff nodes")
	print(len(set(nodes_g1) ^ set(nodes_g2)))
	print("len union edges")
	print(len(set(edges_g1) & set(edges_g2)))
	print("len union nodes")
	print(len(set(nodes_g1) & set(nodes_g2)))
	print("g1 only nodes")
	print(len(nodes_g1_only))
	print("g2 only nodes")
	print(len(nodes_g2_only))
	print("g1 only edges")
	print(len(edges_g1_only))
	print("g2 only edges")
	print(len(edges_g2_only))
	print(ig.summary(graph_1))
	print(ig.summary(graph_2))
	for x in nodes_g1_only:
		if x in nodes_g2:
			print("error")
		if x not in nodes_g1:
			print("error")
	for x in nodes_g2_only:
		if x in nodes_g1:
			print("error")
		if x not in nodes_g2:
			print("error")
	for x in nodes_common:
		if x not in nodes_g1:
			print("error")
		if x not in nodes_g2:
			print("error")
	for x in edges_g1_only:
		if x in edges_g2:
			print("error")
		if x not in edges_g1:
			print("error")
	for x in edges_g2_only:
		if x in edges_g1:
			print("error")
		if x not in edges_g2:
			print("error")
	for x in edges_common:
		if x not in edges_g1:
			print("error")
		if x not in edges_g2:
			print("error")
	return nodes_g1, edges_g1, nodes_g1_only, edges_g1_only, \
	nodes_g2, edges_g2, nodes_g2_only, edges_g2_only, \
	nodes_common, edges_common