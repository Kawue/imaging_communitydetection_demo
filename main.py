from __future__ import print_function, division

from sys import argv
from kode.workflow_barley_101 import *
from kode.workflow_glioblastoma_N3378_106 import *
from kode.workflow_2D_gaussians import *
from kode.workflow_custom import *

if __name__ == '__main__':
	dataset = {1: "barley_101",
			   2: "glioblastoma_N3378_106",
			   3: "2D_gaussians",
			   4: "Custom data set"}

	selected_dataset = int(argv[1])

	if selected_dataset == 1:
		workflow_barley_101()
	elif selected_dataset == 2:
		workflow_glioblastoma_N3378_106()
	elif selected_dataset == 3:
		workflow_2D_gaussians()
	elif selected_dataset == 4:
		workflow_custom()
	else:
		raise ValueError("Wrong Dataset ID!")