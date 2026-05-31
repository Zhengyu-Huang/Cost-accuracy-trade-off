import os
import torch
import sys
import argparse

from pathlib import Path

import numpy as np
from timeit import default_timer

from mpcno_helper import gen_data_tensors

sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from nn.mpcno import compute_Fourier_modes, MPCNO, MPCNO_train

torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model with different configurations and options.')

    parser.add_argument('--grad', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--geo', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--geointegral', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--k_max', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--n_train', type=int, default=900)
    parser.add_argument('--n_test', type=int, default=100)
    parser.add_argument('--act', type=str, default="gelu")
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--dx_scale', type=float, default=1.0)


    args = parser.parse_args()

    layer_selection = {'grad': args.grad.lower() == "true", 'geo': args.geo.lower() == "true", 'geointegral': args.geointegral.lower() == "true"}
    f_in_dim = 0
    f_out_dim = 1

    k_max = args.k_max
    ndim = 3
    n_layer = args.n_layer
    layers = [64]*(n_layer+1)
    act = args.act
    dx_scale = args.dx_scale
    mesh_type = args.mesh_type
    n_train = args.n_train
    n_test  = args.n_test
    
    save_model_name = f"models/MNO_model_N{n_train}_k{k_max}_nlayer{n_layer}"
    
    ###################################
    # load data
    ###################################
    data_path = "../../data/aerodynamics/PressureVTK_Processed"
 


        
    # load data n_train + n_test
    data = np.load(data_path+"/mpcno_data_n_train"+str(n_train)+"_n_test"+str(n_test)+".npz")
    names_array = np.load(data_path+"/mpcno_data_names_list"+"_n_train"+str(n_train)+"_n_test"+str(n_test)+".npy", allow_pickle=True)
    

    nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
    print(nnodes.shape,node_mask.shape,nodes.shape,flush = True)
    
    #！！！！！
    Ls = [10.0, 4.0, 3.2]

    # TODO single measure 
    node_weights = data["node_measures"]
    node_weight_scale = np.amax(np.sum(node_weights, axis=1))
    node_weights = node_weights / node_weight_scale  
    # node_weights = compute_unnormalized_node_measures(nnodes, node_weights, [2], Ls):
    
    node_weights = node_weights[...,0]

    directed_edges, edge_gradient_weights = data["directed_edges"], data["edge_gradient_weights"] / dx_scale
    features = data["features"]

    print(args)
    ndata = nodes.shape[0]
    assert(ndata == n_train + n_test)
    print(f"ndata: {ndata},  n_train: {n_train}, n_test: {n_test}", flush=True)
        




    print("Casting to tensor", flush=True)
    nnodes = torch.from_numpy(nnodes)
    node_mask = torch.from_numpy(node_mask)
    nodes = torch.from_numpy(nodes.astype(np.float32))
    node_weights = torch.from_numpy(node_weights.astype(np.float32))
    features = torch.from_numpy(features.astype(np.float32))
    directed_edges = torch.from_numpy(directed_edges.astype(np.int64))
    edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32)) 


    x_train, y_train, aux_train = gen_data_tensors(np.arange(n_train), nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights, f_in_dim, f_out_dim)
    x_test, y_test, aux_test = gen_data_tensors(np.arange(-n_test, 0), nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights, f_in_dim, f_out_dim)





    print(f'x_train shape {x_train.shape}, x_test shape {x_train.shape}, y_train shape {y_train.shape}, y_test shape {y_train.shape}', flush = True)
    print('length of each dim: ',torch.amax(nodes, dim = [0,1]) - torch.amin(nodes, dim = [0,1]), flush = True)



    print(f'kmax = {k_max}')
    print(f'n_train = {n_train}, n_test = {n_test}')
    print(f'Ls = {Ls}')
    print(f'layer_selection = {layer_selection}')
    print(f'layers = {layers}')
    print(f'activation = {act}')


    modes = compute_Fourier_modes(ndim, [k_max, k_max, k_max], Ls)
    modes = torch.tensor(modes, dtype=torch.float).to(device)
    model = MPCNO(ndim, modes,
                layer_selection = layer_selection,
                layers=layers,
                fc_dim=64,
                in_dim=x_train.shape[-1], out_dim=y_train.shape[-1],
                act = act,
                ).to(device)



    epochs = args.epochs
    base_lr = 5e-4
    lr_ratio = 10
    scheduler = "OneCycleLR"
    weight_decay = 1.0e-4
    batch_size = args.batch_size
    print(f'batch_size = {batch_size}')

    normalization_x = False
    normalization_y = True
    normalization_dim_x = []
    normalization_dim_y = []
    non_normalized_dim_x = 4
    non_normalized_dim_y = 0


    config = {"train" : {"base_lr": base_lr, 'lr_ratio': lr_ratio, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
                        "normalization_x": normalization_x,"normalization_y": normalization_y, 
                        "normalization_dim_x": normalization_dim_x, "normalization_dim_y": normalization_dim_y, 
                        "non_normalized_dim_x": non_normalized_dim_x, "non_normalized_dim_y": non_normalized_dim_y}
                        }


    train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = MPCNO_train(
        x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name=save_model_name)
