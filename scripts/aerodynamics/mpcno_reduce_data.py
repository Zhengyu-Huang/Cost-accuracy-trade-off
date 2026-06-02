import os
import argparse
import numpy as np

from mpcno_helper import (
    DrivAerNet_datasets,
    random_shuffle
)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model with different configurations and options.')

    parser.add_argument('--n_train', type=int, default=900)
    parser.add_argument('--n_test', type=int, default=100)
    
    args = parser.parse_args()


    n_train = args.n_train
    n_test  = args.n_test
    mesh_type = 'vertex_centered'


    ###################################
    # load data
    ###################################
    data_path = "../../data/aerodynamics/PressureVTK_Processed"
 
    # load data n_train + n_test
    equal_weights = False
    data = np.load(data_path+"/mpcno_data.npz")
    names_array = np.load(data_path+"/mpcno_data_names_list.npy", allow_pickle=True)
    
    # random shuffle, and keep only n_train + n_test data
    data, names_list = random_shuffle(data, names_array, n_train, n_test, seed=42)
    
    print("max nnodes = ", np.max(data["nnodes"]))    
    np.savez(data_path+"/mpcno_data_n_train"+str(n_train)+"_n_test"+str(n_test)+".npz", **data)

    # np.savez_compressed(data_path+"/mpcno_data_n_train"+str(n_train)+"_n_test"+str(n_test)+".npz", \
    #                     nnodes=data["nnodes"], node_mask=data["node_mask"], nodes=data["nodes"], \
    #                     node_measures=data["node_measures"], \
    #                     features=data["features"], \
    #                     directed_edges=data["directed_edges"], edge_gradient_weights=data["edge_gradient_weights"]) 

    np.save(os.path.join(data_path, "mpcno_data_names_list"+"_n_train"+str(n_train)+"_n_test"+str(n_test)+".npy"), names_list)



