import meshio
import torch
import os
import sys
import numpy as np
import gc
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from matplotlib.ticker import ScalarFormatter
from mpcno_helper import gen_data_tensors

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utility.normalizer import UnitGaussianNormalizer
from utility.losses import LpLoss
from nn.mpcno import compute_Fourier_modes, MPCNO
from nn.geo_utility import compute_node_weight_scale


plt.style.use('seaborn-v0_8-whitegrid')   # 现代网格样式
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 28,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2.4
})
formatter = ScalarFormatter(useMathText=True)
# 2. 强制使用科学计数法，并让指数作为偏移量（顶部显示）
formatter.set_scientific(True)
formatter.set_powerlimits((-2, 2))   # 数值小于 1e-3 或大于 1e3 时触发偏移量
formatter.set_useOffset(True)        # 明确使用偏移量    
lbl = "#000000"
tk = "#808080"



def get_median_index(arr):
    # 确保输入是一个 NumPy 数组
    arr = np.asarray(arr)
    # 获取排序后的索引
    sorted_indices = np.argsort(arr)
    # 计算中位数的索引
    mid_index = len(arr) // 2
    
    if len(arr) % 2 == 1:
        # 如果是奇数长度，返回中间元素的原始索引
        median_index = sorted_indices[mid_index]
    else:
        # 如果是偶数长度，返回中间两个元素的原始索引
        median_index_1 = sorted_indices[mid_index - 1]
        median_index_2 = sorted_indices[mid_index]
        # 通常我们不会为偶数长度的数组返回单个索引，因为中位数是两个值的平均。
        # 但是，如果你需要，你可以选择返回这两个索引或仅其中一个。
        # 这里我们简单地返回一个元组
        median_index = [median_index_1, median_index_2]
    
    return median_index




def plot_results(vertices, elems, vertex_data, elem_data, file_name):
    
    """
    Save mesh data to a VTK file using meshio.
    
    Parameters:
    -----------
    vertices : numpy array (n_vertices, 3)
        Vertex coordinates
    elems : numpy array (n_elements, 4)
        Element connectivity (assuming triangle elements with 3 nodes + 1 element dimension)
    vertex_data : dict
        Data defined at vertices (e.g., {"displacement": vertex_displacements})
    elem_data : dict
        Data defined at elements (e.g., {"stress": element_stresses})
    file_name : str
        Output file name without extension
    """
 
    cells = []
    cells.append(("triangle", elems[:,1:]))

    # Create the mesh
    mesh = meshio.Mesh(
        points=vertices,
        cells=cells,
        point_data=vertex_data,
        cell_data=elem_data,
    )

    # Save to an Exodus II file
    meshio.write(file_name+ ".vtk", mesh)




def predict_error(data_path, n_train, n_test, data_ids = None):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    ##############################################
    # load data
    ##############################################
    data = np.load(data_path+"/mpcno_data_n_train"+str(n_train)+"_n_test"+str(n_test)+".npz")
    
    names_array = np.load(data_path+"/mpcno_data_names_list"+"_n_train"+str(n_train)+"_n_test"+str(n_test)+".npy", allow_pickle=True)
    
    
    dx_scale = 10.0
    k_max = 16
    n_layer = 4
    fc_dim = 64
    layers = [fc_dim]*(n_layer+1)
    layer_selection = {'grad': True, 'geo': True, 'geointegral': True}
    n_point = int(data_path.split('_')[-1])
    #！！！！！
    # bounding box [5.2715902328491211, 2.3783199787139893, 1.7617900371551514]
    Ls = [10.0, 4.0, 3.2]
    # Ls = [7.0, 3.0, 2.0]

    
    normalization_x = False
    normalization_y = True

    save_model_name = f"models/MPCNO_model_N{n_train}_k{k_max}_nlayer{n_layer}_npoint{n_point}"
    
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
    

    


    k_max = 16
    ndim = 3
    modes = compute_Fourier_modes(ndim, [k_max, k_max, k_max], Ls)
    modes = torch.tensor(modes, dtype=torch.float).to(device)
    model = MPCNO(ndim, modes, 
                layer_selection = layer_selection,
                layers=layers,
                fc_dim=fc_dim,
                in_dim=x_train.shape[-1], out_dim=y_train.shape[-1],
                act = 'gelu',
                ).to(device)
    
    model.load_state_dict(torch.load(save_model_name+".pth", map_location="cpu"))
    model = model.to(device)
    
    

    
    x_normalizer = UnitGaussianNormalizer.from_state_dict(torch.load(save_model_name + "_normalization_x.pth", map_location="cpu", weights_only=True,), device=device) if normalization_x else None
    y_normalizer = UnitGaussianNormalizer.from_state_dict(torch.load(save_model_name + "_normalization_y.pth", map_location="cpu", weights_only=True,), device=device) if normalization_y else None
      


    myloss = LpLoss(d=1, p=2, size_average=False)
    rl1loss = LpLoss(d=1, p=1, size_average=False)

    if data_ids is None:
        test_rel_l2 = np.zeros(n_test)
        test_rel_l1 = np.zeros(n_test)
        for i in range(n_test):
            x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = x_test[[i],...], y_test[[i],...], aux_test[0][[i],...], aux_test[1][[i],...], aux_test[2][[i],...], aux_test[3][[i],...], aux_test[4][[i],...], aux_test[5][[i],...]
            x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device), geo.to(device)

            batch_size_ = x.shape[0]
            out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo)) #.reshape(batch_size_,  -1)

            if normalization_y:
                out = y_normalizer.decode(out)
                # y = y_normalizer.decode(y)
            out=out*node_mask #mask the padded value with 0,(1 for node, 0 for padding)
            test_rel_l2[i] = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
            test_rel_l1[i] = rl1loss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
            print("Test ", names_array[n_train+i] , " rel. L2 error ", test_rel_l2[i], " rel. L1 error ", test_rel_l1[i])

        np.save(f'data/test_rel_l2_npoint{n_point}.npy', test_rel_l2)
        np.save(f'data/test_rel_l1_npoint{n_point}.npy', test_rel_l1)
    
        largest_rl1_error_ind = np.argmax(test_rel_l1)
        largest_2nd_rl1_error_ind = np.argsort(test_rel_l1)[-2]
        median_1st_rl1_error_ind, median_2nd_rl1_error_ind = get_median_index(test_rel_l1)  # Get the index (or indices)
        print("largest rel. L1 error is ", test_rel_l1[largest_rl1_error_ind], " ; median rel. L1 error is ", test_rel_l1[median_1st_rl1_error_ind], test_rel_l1[median_2nd_rl1_error_ind])
        print("largest rel. L1 error index is ", largest_rl1_error_ind, " ; median rel. L1 error index is ", median_1st_rl1_error_ind, median_2nd_rl1_error_ind)
        # they are only test data id
        data_ids = [largest_rl1_error_ind + n_train, largest_2nd_rl1_error_ind+n_train, median_1st_rl1_error_ind + n_train, median_2nd_rl1_error_ind + n_train]
        
        
        
        
    for data_id in data_ids:
        
        ################################################
        # load raw data to get elems, vertices
        ################################################
        _, raw_data_id = names_array[data_id].split("/")
        vtk_file = os.path.join(data_path, names_array[data_id])
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(vtk_file)
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


        
        
        if data_id < n_train:
            x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = (x_train[[data_id],...], y_train[[data_id],...], aux_train[0][[data_id],...], aux_train[1][[data_id],...], aux_train[2][[data_id],...], aux_train[3][[data_id],...], aux_train[4][[data_id],...], aux_train[5][[data_id],...])
        else:
            x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = (x_test[[data_id - n_train],...], y_test[[data_id - n_train],...], aux_test[0][[data_id - n_train],...], aux_test[1][[data_id - n_train],...], aux_test[2][[data_id - n_train],...], aux_test[3][[data_id - n_train],...], aux_test[4][[data_id - n_train],...], aux_test[5][[data_id - n_train],...])
        
        x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device), geo.to(device)

        batch_size_ = x.shape[0]
        out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo)) #.reshape(batch_size_,  -1)
        if normalization_y:
            out = y_normalizer.decode(out)
            # y = y_normalizer.decode(y)
        out=out*node_mask #mask the padded value with 0,(1 for node, 0 for padding)
        print(vtk_file, " rel. L2 error ",  myloss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item(), 
                        " rel. L1 error ", rl1loss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item())
        

       
        
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
        
        file_name = "figs/predict_npoint" + str(n_point) + "_" + raw_data_id
        print("file name is ", file_name)
        meshio.write(file_name, mesh)


def error_bin_plot(n_point):
    data = np.load(f'data/test_rel_l1_npoint{n_point}.npy')  
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    plt.xlabel(rf"Rel. $L^1$ error")
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Test Relative $L_1$ Errors')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"figs/error_bin_npoint{n_point}.pdf")
        
if __name__ == "__main__":
    n_point = 20000
    predict_error(data_path = f"../../data/aerodynamics/PressureVTK_Processed_{n_point}",  n_train = 4000, n_test = 512, data_ids = None)
    error_bin_plot(n_point)

    n_point = 40000
    predict_error(data_path = f"../../data/aerodynamics/PressureVTK_Processed_{n_point}",  n_train = 4000, n_test = 512, data_ids = None)
    error_bin_plot(n_point)

