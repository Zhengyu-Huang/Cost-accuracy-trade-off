"""Create the seeded training/test subset used by the M-PCNO experiments.

Run from ``scripts/aerodynamics`` after preprocessing one resolution::

    python mpcno_reduce_data.py --n_train 4000 --n_test 512 --n_point 10000

The active ``__main__`` reads ``mpcno_data.npz`` and its aligned names array
from ``../../data/aerodynamics/PressureVTK_Processed_<N>``, applies the fixed
seed-42 selection, and writes the corresponding ``n_train4000_n_test512`` data
and names archives. Repeat for 20000 and 40000 when those models are required.
"""

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
    parser.add_argument('--n_point', type=int, default=10000)
    
    args = parser.parse_args()


    n_train = args.n_train
    n_test  = args.n_test
    n_point  = args.n_point
    mesh_type = 'vertex_centered'


    # Each point resolution is reduced independently from its full archive.
    data_path = "../../data/aerodynamics/PressureVTK_Processed_" + str(n_point)
 
    # The names array is kept aligned with every tensor through the same indices.
    equal_weights = False
    data = np.load(data_path+"/mpcno_data.npz")
    names_array = np.load(data_path+"/mpcno_data_names_list.npy", allow_pickle=True)
    
    # Reorder the retained samples as one training block followed by one test block.
    data, names_list = random_shuffle(data, names_array, n_train, n_test, seed=42)
    
    print("max nnodes = ", np.max(data["nnodes"]))    
    np.savez(data_path+"/mpcno_data_n_train"+str(n_train)+"_n_test"+str(n_test)+".npz", **data)

    np.save(os.path.join(data_path, "mpcno_data_names_list"+"_n_train"+str(n_train)+"_n_test"+str(n_test)+".npy"), names_list)

