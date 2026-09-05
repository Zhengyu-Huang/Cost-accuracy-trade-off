import meshio
import torch
import time
import os
import sys
import numpy as np
import gc
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import re

from mpcno_helper import _load_data, gen_data_tensors

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utility.normalizer import UnitGaussianNormalizer
from utility.losses import LpLoss
from nn.geo_utility import preprocess_data_mesh, compute_node_weight_scale
from nn.mpcno import compute_Fourier_modes, MPCNO, mpcno_floating_point_cost





def mpcno_solver(save_model_name, data_vtk_file):


    ##############################################
    # setup solver
    ##############################################
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    dx_scale = 10.0
    k_max = 16
    n_layer = 4
    fc_dim = 64
    layers = [fc_dim]*(n_layer+1)
    layer_selection = {'grad': True, 'geo': True, 'geointegral': True}

    #！！！！！
    # bounding box [5.2715902328491211, 2.3783199787139893, 1.7617900371551514]
    Ls = [10.0, 4.0, 3.2]
    # Ls = [7.0, 3.0, 2.0]

    
    normalization_x = False
    normalization_y = True

    f_in_dim, f_out_dim = 0, 1
    
    k_max = 16
    ndim  = 3
    modes = compute_Fourier_modes(ndim, [k_max, k_max, k_max], Ls)
    modes = torch.tensor(modes, dtype=torch.float).to(device)
    model = MPCNO(ndim, modes, 
                layer_selection = layer_selection,
                layers=layers,
                fc_dim=fc_dim,
                in_dim=f_in_dim+2*ndim, out_dim=f_out_dim,
                act = 'gelu',
                ).to(device)
    
    model.load_state_dict(torch.load(save_model_name+".pth", map_location="cpu"))
    model = model.to(device)
    
    x_normalizer = UnitGaussianNormalizer.from_state_dict(torch.load(save_model_name + "_normalization_x.pth", map_location="cpu", weights_only=True,), device=device) if normalization_x else None
    y_normalizer = UnitGaussianNormalizer.from_state_dict(torch.load(save_model_name + "_normalization_y.pth", map_location="cpu", weights_only=True,), device=device) if normalization_y else None
      
    ##############################################
    # downsample data and generate data
    ##############################################
    
    names_list, nodes_list, elems_list, features_list = [], [], [], []
    names_list.append(data_vtk_file)
    _load_data(data_vtk_file, nodes_list, elems_list, features_list)
    


    print("Preprocessing data")
    nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data_mesh(nodes_list, elems_list, features_list, mesh_type = 'vertex_centered', adjacent_type="edge")
    

    node_weights = np.nan_to_num(node_measures_raw, nan=0.0)
    node_weight_scale = compute_node_weight_scale(2, Ls)
    node_weights = node_weights / node_weight_scale  
    node_weights = node_weights[...,0]
    edge_gradient_weights /= dx_scale
       


    print("Casting to tensor", flush=True)
    nnodes = torch.from_numpy(nnodes)
    node_mask = torch.from_numpy(node_mask)
    nodes = torch.from_numpy(nodes.astype(np.float32))
    node_weights = torch.from_numpy(node_weights.astype(np.float32))
    features = torch.from_numpy(features.astype(np.float32))
    directed_edges = torch.from_numpy(directed_edges.astype(np.int64))
    edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))


    x_test, y_test, aux_test = gen_data_tensors(np.arange(1), nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights, f_in_dim, f_out_dim)
    
    

    print(f'x_test shape {x_test.shape}, y_test shape {y_test.shape}', flush = True)
    print('length of each dim: ',torch.amax(nodes, dim = [0,1]) - torch.amin(nodes, dim = [0,1]), flush = True)
    print(f'kmax = {k_max}')
    print(f'Ls = {Ls}')
    print(f'layer_selection = {layer_selection}')
    print(f'layers = {layers}')
    






    myloss = LpLoss(d=1, p=2, size_average=False)
    rl1loss = LpLoss(d=1, p=1, size_average=False)

    
        
    
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(data_vtk_file)
    reader.Update()
    polydata = reader.GetOutput()
    # coordinates
    points = polydata.GetPoints()
    num_points = points.GetNumberOfPoints()
    vertices = np.array([points.GetPoint(i) for i in range(num_points)])
    # elements（假设全是三角形）
    polys = polydata.GetPolys()   # vtkCellArray
    # 转换为 numpy 数组
    cell_array = vtk_to_numpy(polys.GetData())
    # 解析：数组结构为 [3, id0, id1, id2, 3, id0, id1, id2, ...]
    elems = cell_array.reshape(-1, 4)[:,1:] 


        
    x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = x_test, y_test, aux_test[0], aux_test[1], aux_test[2], aux_test[3], aux_test[4], aux_test[5]
    x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device), geo.to(device)

    batch_size_ = x.shape[0]
    out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo)) #.reshape(batch_size_,  -1)

    if normalization_y:
        out = y_normalizer.decode(out)
        # y = y_normalizer.decode(y)
    out=out*node_mask #mask the padded value with 0,(1 for node, 0 for padding)
    test_rel_l2 = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
    test_rel_l1 = rl1loss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
    print("Test  rel. L2 error ", test_rel_l2, " rel. L1 error ", test_rel_l1)

      
       
        
    node_mask_bool = node_mask[0,:,0].bool().numpy()
    y = y.cpu().detach().numpy()[0, node_mask_bool ,0]
    out = out.cpu().detach().numpy()[0, node_mask_bool ,0]
        
    # Nodal values: scalar (e.g., temperature) and vector (e.g., displacement)
    Cp_ref = y  # Scalar values at each node
    Cp_pred = out  # Scalar values at each node
    Cp_error = out - y  # Scalar values at each node
    # Convert nodes to meshio-compatible format
        

    # Convert elements to meshio-compatible format
    cells = []
    cells.append(("triangle", elems))

    # Create the mesh
    mesh = meshio.Mesh(
        points=vertices,
        cells=cells,
        point_data={
            "Cp_ref": Cp_ref,
            "Cp_pred": Cp_pred,
            "Cp_error": Cp_error,
        }
    )
    
    
    idx = data_vtk_file.rfind("/")
    file_name = data_vtk_file[:idx+1] + "predicted_" + data_vtk_file[idx+1:]
    print("save data file: ", file_name)
    meshio.write(file_name, mesh)
        





def cost_accuracy_mpcno_solver_helper(device, data_path, save_model_name, n_train, n_test, n_trial):

    ##############################################
    # load data
    ##############################################
    data = np.load(data_path+"/mpcno_data_n_train"+str(n_train)+"_n_test"+str(n_test)+".npz")
    
    names_array = np.load(data_path+"/mpcno_data_names_list"+"_n_train"+str(n_train)+"_n_test"+str(n_test)+".npy", allow_pickle=True)
    

    k_max = 16 
    
    n_train = int(re.search(r'N(\d+)', save_model_name).group(1))
    n_layer = int(re.search(r'nlayer(\d+)', save_model_name).group(1))

    dx_scale = 10.0
    fc_dim = 64
    layers = [fc_dim]*(n_layer+1)
    layer_selection = {'grad': True, 'geo': True, 'geointegral': True}

    #！！！！！
    # bounding box [5.2715902328491211, 2.3783199787139893, 1.7617900371551514]
    Ls = [10.0, 4.0, 3.2]
    # Ls = [7.0, 3.0, 2.0]

    
    normalization_x = False
    normalization_y = True

    
    f_in_dim, f_out_dim = 0, 1
    nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
    print(nnodes.shape,node_mask.shape,nodes.shape,flush = True)
    
    node_weights = data["node_measures"]
    node_weight_scale = compute_node_weight_scale(2, Ls)
    node_weights = node_weights / node_weight_scale  
    
    node_weights = node_weights[...,0]


    directed_edges, edge_gradient_weights = data["directed_edges"], data["edge_gradient_weights"] / dx_scale
    features = data["features"]

    ndata = nodes.shape[0]
    assert(ndata == n_train + n_test)

    # delete data and release its memory
    del data
    gc.collect()


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



    print(f'x_train shape {x_train.shape}, x_test shape {x_test.shape}, y_train shape {y_train.shape}, y_test shape {y_train.shape}', flush = True)
    print('length of each dim: ',torch.amax(nodes, dim = [0,1]) - torch.amin(nodes, dim = [0,1]), flush = True)
    print(f'kmax = {k_max}')
    print(f'n_train = {n_train}, n_test = {n_test}')
    print(f'Ls = {Ls}')
    print(f'layer_selection = {layer_selection}')
    print(f'layers = {layers}')
    

    ndim = 3
    modes = compute_Fourier_modes(ndim, [k_max, k_max, k_max], Ls)
    modes = torch.tensor(modes, dtype=torch.float).to(device)
    model = MPCNO(ndim, modes, 
                layer_selection = layer_selection,
                layers=layers,
                fc_dim=fc_dim,
                in_dim=f_in_dim+2*ndim, out_dim=y_train.shape[-1],
                act = 'gelu',
                ).to(device)
    
    model.load_state_dict(torch.load(save_model_name+".pth", map_location="cpu"))
    model = model.to(device)
    
    x_normalizer = UnitGaussianNormalizer.from_state_dict(torch.load(save_model_name + "_normalization_x.pth", map_location="cpu", weights_only=True,), device=device) if normalization_x else None
    y_normalizer = UnitGaussianNormalizer.from_state_dict(torch.load(save_model_name + "_normalization_y.pth", map_location="cpu", weights_only=True,), device=device) if normalization_y else None
      
    myloss = LpLoss(d=1, p=2, size_average=False)
    rl1loss = LpLoss(d=1, p=1, size_average=False)


    n_repeat = 10
    
    
    accuracy = np.zeros((n_trial, 2)) # rel l2 err, rel l1 err
    cost = np.zeros((n_trial, 2))     # floating point cost, run time 
    
    for i in range(n_trial):
        x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = x_test[[i],...], y_test[[i],...], aux_test[0][[i],...], aux_test[1][[i],...], aux_test[2][[i],...], aux_test[3][[i],...], aux_test[4][[i],...], aux_test[5][[i],...]
        x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device), geo.to(device)

        batch_size_ = x.shape[0]
        # warm-up
        out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo)) #.reshape(batch_size_,  -1)


        start_time = time.perf_counter()
        for j in range(n_repeat):
            out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo)) #.reshape(batch_size_,  -1)

            if normalization_y:
                out = y_normalizer.decode(out)
                # y = y_normalizer.decode(y)
            out=out*node_mask #mask the padded value with 0,(1 for node, 0 for padding)
            
        end_time = time.perf_counter()
        solve_time = (end_time - start_time)/n_repeat
        
        accuracy[i,0] = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
        accuracy[i,1] = rl1loss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
        cost[i,0] = mpcno_floating_point_cost(ndim, f_in_dim+2*ndim, f_out_dim, k_max, fc_dim, n_layer, node_mask.sum().item(), layer_selection=layer_selection)
        cost[i,1] = solve_time

                    
                    
        print("Test ", names_array[n_train+i] , " rel. L2 error ", accuracy[i,0], " rel. L1 error ", accuracy[i,1], "cost : ", cost[i,1])

       
    return cost, accuracy



def cost_accuracy_mpcno_solver(n_point_values):
    n_train, n_test, n_trial = 4000, 512, 512
    
    n_layer = 4
    cost_array = np.zeros((len(n_point_values), n_trial, 3)) 
    accuracy_array = np.zeros((len(n_point_values),  n_trial, 2))
    
        
    for n_point_index, n_point in enumerate(n_point_values):
        data_path = "../../data/aerodynamics/PressureVTK_Processed_" + str(n_point)
        save_model_name = f"models/MPCNO_model_N{n_train}_k16_nlayer{n_layer}_npoint{n_point}"
        for device in [torch.device('cpu') , torch.device('cuda')]:
            
            cost, accuracy  = cost_accuracy_mpcno_solver_helper(device, data_path, save_model_name=save_model_name, n_train = n_train, n_test = n_test, n_trial = n_trial)
            print("n_point is ", n_point, " cost is ", cost)
            
            cost_array[n_point_index, ..., 0] = cost[...,0]
            if device.type == 'cpu':
                cost_array[n_point_index, ..., 1] = cost[...,1]
            else:
                cost_array[n_point_index, ..., 2] = cost[...,1]
                
            accuracy_array[n_point_index, ...] = accuracy
             
                

    np.savez_compressed('data/cost_accuracy_mpcno_solver_data.npz', cost=cost_array, accuracy=accuracy_array)

    return  cost_array, accuracy_array 


if __name__ == "__main__":
    # save_model_name =  "models/MPCNO_model_N2000_k16_nlayer4_npoint10000"
    # # data_vtk_file = "data/openfoam_L_decimate.vtk"
    # data_vtk_file = "data/F_D_WM_WW_0057.vtk"
    # mpcno_solver(save_model_name, data_vtk_file)
    
    cost_accuracy_mpcno_solver([10000, 20000, 40000])
    
