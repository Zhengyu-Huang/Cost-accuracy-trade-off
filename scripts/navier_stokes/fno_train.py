import sys
import os
import math 
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 向上两级找到项目根目录
project_root = os.path.dirname(os.path.dirname(current_dir))
# 添加到路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from baselines.fno import FNO2d, FNO_train



def preprocess_data(n_train, n_test, downsample = 1):
    '''
    参数：
    n_train: 训练样本数量
    n_test: 测试样本数量
    '''
    # the data is on 256 by 256 array 
    n_total_data, n, nt = 10, 256, 50
    stride = 2**downsample 
    
    n = n // stride
    
    L = 1.0
    
    x = np.linspace(0.0, L, n, endpoint=False)
    y = np.linspace(0.0, L, n, endpoint=False)
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    
    X, Y = [], []
    for i in list(range(math.ceil(n_train / nt))) + [n_total_data + x for x in range(-math.ceil(n_test / nt), 0)]:
        data = np.load(f"../../data/navier_stokes/navier_stokes_{i:05d}.npy")
        # data : nt+2 by n by n array. 
        # forcing, vorticity_0, vorticity_1, ......, vorticity_nt
        vorticity = data[1:, ::stride, 0::stride]
        force     = data[0, ::stride, 0::stride]
        for j in range(nt):
            X.append(np.stack([vorticity[j,:,:], force, x_grid, y_grid], axis=2))         # omega, force, x, y
            Y.append(vorticity[j+1,:,:,np.newaxis])                                       # omega'
            
    X, Y = np.array(X), np.array(Y)
    X, Y = torch.from_numpy(X.astype(np.float32)), torch.from_numpy(Y.astype(np.float32))
    x_train, y_train  = X[:n_train,...], Y[:n_train,...] 
    x_test,  y_test   = X[-n_test:,...], Y[-n_test:,...] 

    print("x_train.shape = ", x_train.shape, "y_train.shape = ", y_train.shape)
    
    return x_train, y_train, x_test, y_test


def setup_model(in_dim, out_dim, device, checkpoint_path = None):
    nlayers = 5
    model = FNO2d([16]*nlayers, [16]*nlayers,
                layers=[64]*nlayers,
                fc_dim=64,
                in_dim=in_dim, out_dim=out_dim,
                act='gelu',
                pad_ratio=0.05, 
                cnn_kernel_size=1)
    
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    model = model.to(device)

    return model

if __name__ == "__main__":

    in_dim, out_dim = 4, 1

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = setup_model(in_dim, out_dim, device)

    n_train, n_test = 200, 200
    downsample = 2
    x_train, y_train, x_test, y_test = preprocess_data(n_train, n_test, downsample)

    epochs = 500
    base_lr = 5e-4 #0.001
    lr_ratio = 10
    scheduler = "OneCycleLR"
    weight_decay = 1.0e-4
    batch_size = 8

    print('batch_size', batch_size, '\n')

    normalization_x = True
    normalization_y = True
    normalization_dim_x = [0,1] #channel-wise normalization
    normalization_dim_y = []    #normalization
    non_normalized_dim_x = 0
    non_normalized_dim_y = 0

    config = {"train" : {"base_lr": base_lr, 'lr_ratio': lr_ratio, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
                        "normalization_x": normalization_x,"normalization_y": normalization_y, 
                        "normalization_dim_x": normalization_dim_x, "normalization_dim_y": normalization_dim_y, 
                        "non_normalized_dim_x": non_normalized_dim_x, "non_normalized_dim_y": non_normalized_dim_y}
                        }


    train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = FNO_train(x_train, y_train, x_test, y_test, config, model, save_model_name="./FNO_model")
    

    



