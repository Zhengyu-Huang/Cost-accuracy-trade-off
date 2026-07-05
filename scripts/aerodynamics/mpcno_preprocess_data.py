import os
import torch
import argparse
import sys
import numpy as np
from pathlib import Path

from mpcno_helper import (
    DrivAerNet_datasets,
    load_data,
)

sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from nn.geo_utility import preprocess_data_mesh



torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model with different configurations and options.')
    # Preprocess data n_each from each subcategories
    parser.add_argument('--n_each', type=int, default=100)
    parser.add_argument('--n_point', type=int, default=10000)
    
    # Specifies how the computational mesh is represented. 
    # “cell_centered” stores features at cell centers (control-volume based), 
    # while “vertex_centered” stores features at mesh vertices (node-based).
    args = parser.parse_args()
    mesh_type = 'vertex_centered'
    n_each  = args.n_each 
    n_point = args.n_point
    ###################################
    # load data
    ###################################
    data_path = "../../data/aerodynamics/PressureVTK_Processed_" + str(n_point)
    
                                                                                                

    print("Loading data: ", n_each, " from each datasets")
    print("DrivAerNet datasets: ", DrivAerNet_datasets)
    
    nodes_list, elems_list, features_list, names_list =  load_data(data_path, DrivAerNet_datasets, n_each)
    ndata = len(nodes_list)
    
    print("Preprocessing data")
    nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data_mesh(nodes_list, elems_list, features_list, mesh_type = mesh_type, adjacent_type="edge")
    node_measures = np.nan_to_num(node_measures_raw, nan=0.0)
    np.savez_compressed(data_path+"/mpcno_data.npz", \
                        nnodes=nnodes, node_mask=node_mask, nodes=nodes, \
                        node_measures=node_measures, \
                        features=features, \
                        directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights) 
    
    np.save(os.path.join(data_path, "mpcno_data_names_list.npy"), np.array(names_list, dtype=object))

    
