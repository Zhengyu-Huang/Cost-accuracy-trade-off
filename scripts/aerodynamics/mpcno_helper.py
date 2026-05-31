import os
import numpy as np
import torch
import vtk
from vtk.util.numpy_support import vtk_to_numpy

DrivAerNet_datasets = [
    "E_S_WW_WM",
    "E_S_WWC_WM",
    "F_D_WM_WW_1",
    "F_D_WM_WW_2",
    "F_D_WM_WW_3",
    "F_D_WM_WW_4",
    "F_D_WM_WW_5",
    "F_D_WM_WW_6",
    "F_D_WM_WW_7",
    "F_D_WM_WW_8",
    "F_S_WWC_WM",
    "F_S_WWS_WM",
    "N_S_WW_WM",
    "N_S_WWC_WM",
    "N_S_WWS_WM"
]


def _load_data(vtk_file, nodes_list, elems_list, elem_features_list):
    """
    使用 vtk.vtkPolyDataReader 读取 .vtk 文件，并保存为 .ply 格式, 只保存了网格信息
    """
    print(f"正在读取 VTK 文件: {vtk_file}")
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file)
    reader.Update()

    polydata = reader.GetOutput()

    # coordinates
    points = polydata.GetPoints()
    num_points = points.GetNumberOfPoints()
    nodes = np.array([points.GetPoint(i) for i in range(num_points)])

    # elements（假设全是三角形）
    polys = polydata.GetPolys()   # vtkCellArray
    # 转换为 numpy 数组
    cell_array = vtk_to_numpy(polys.GetData())
    # 解析：数组结构为 [3, id0, id1, id2, 3, id0, id1, id2, ...]
    elems = cell_array.reshape(-1, 4)   
    elems[:,0] = 2  # elem dim is 2


    # pressure
    scalars = polydata.GetPointData().GetArray("p")
    pressure_feature = np.array([scalars.GetValue(i) for i in range(num_points)])
    # TODO u_inf is 30, compute pressure coefficient
    u_inf = 30
    cp_feature = pressure_feature / (0.5*u_inf**2)

    # normals
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(reader.GetOutput())
    # 关键设置：确保计算点法线，这通常是默认开启的
    normals.SetComputePointNormals(True)
    normals.SetComputeCellNormals(False)  # 本例只计算点法线，你也可以根据需要开启
    normals.SetSplitting(False)          # 关键：禁止分裂
    normals.Update()
    point_normals = vtk_to_numpy(normals.GetOutput().GetPointData().GetNormals())
    
    elem_features = np.column_stack((point_normals, cp_feature))  
    
    nodes_list.append(nodes)
    elems_list.append(elems)
    elem_features_list.append(elem_features)


    # 可以这样检查
    # cells = []
    # cells.append(("triangle", elems[:,1:]))
    # vertex_data = {"pressure": elem_features[:,-1], "normals": elem_features[:,0:3]}
    # elem_data = None
    # mesh = meshio.Mesh(
    #     points=nodes,
    #     cells=cells,
    #     point_data=vertex_data,
    #     cell_data=elem_data)
    # meshio.write("test.vtk", mesh)



def load_data(data_path, DrivAerNet_datasets, n_each):
    names_list, nodes_list, elems_list, elem_features_list = [], [], [], []

    DrivAerNet_dir = data_path

        
    for subdir in DrivAerNet_datasets:
        folder_path = os.path.join(DrivAerNet_dir, subdir)
        print("folder_path = ", folder_path)
        for dirpath, _, filenames in os.walk(folder_path):
            print("Load ", n_each, " data from ", folder_path,  ", containing ", len(filenames), " data ")
        
            for i in range(min(n_each,len(filenames))):
                file_path = os.path.join(folder_path, filenames[i])
                _load_data(file_path, nodes_list, elems_list, elem_features_list)
                names_list.append(os.path.join(subdir, filenames[i]))

    return nodes_list, elems_list, elem_features_list, names_list




def random_shuffle(data, names_array, n_train, n_test, seed=42):
    np.random.seed(seed)  # 可选的：为了可重复性设置随机种子
    
    ndata = data["nodes"].shape[0]
    assert(ndata >= n_train + n_test)
    print("Total data number =", ndata, " n_train = ", n_train, " n_test = ", n_test)
    random_indices = np.arange(ndata)
    np.random.shuffle(random_indices)
    
    # 取前n_train 和后n_test 个分别作为训练和测试集
    train_indices = random_indices[:n_train]
    test_indices = random_indices[-n_test:]
    indices = np.concatenate([train_indices, test_indices])
    
    
    data = {key: value[indices] for key, value in data.items()}
    names_array = names_array[indices]
    
    # 输出数据统计情况
    all_datasets = DrivAerNet_datasets
    train_data_stats = {subdir: 0 for subdir in all_datasets}
    test_data_stats = {subdir: 0 for subdir in all_datasets}
    for i in range(n_train):
        train_data_stats[names_array[i].split('/')[-2]] += 1
    for i in range(-n_test,0):
        test_data_stats[names_array[i].split('/')[-2]] += 1
    print("Training data statistics:")
    print("-" * 40)
    assert(sum(train_data_stats.values()) == n_train)
    for dataset in sorted(all_datasets):
        count = train_data_stats[dataset]
        if count > 0:
            percentage = count / n_train * 100
            print(f"  {dataset:15s}: {count:3d} ({percentage:5.1f}%)")
    
    print("Test data statistics:")
    print("-" * 40)
    assert(sum(test_data_stats.values()) == n_test)
    for dataset in sorted(all_datasets):
        count = test_data_stats[dataset]
        if count > 0:
            percentage = count / n_test * 100
            print(f"  {dataset:15s}: {count:3d} ({percentage:5.1f}%)")
    
    return data, names_array


# prepare data
def gen_data_tensors(data_indices, nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights, f_in_dim, f_out_dim):
    nodes_input = nodes.clone()
    ndim = nodes.shape[-1]
    # input x （normal, coordinate）
    x = torch.cat((features[data_indices][...,:f_in_dim+ndim], nodes_input[data_indices, ...]), -1)
    # output y
    y = features[data_indices][...,-f_out_dim:]
    # outward normal
    nx = features[data_indices][...,f_in_dim:f_in_dim+ndim]
    aux = (node_mask[data_indices], nodes[data_indices], node_weights[data_indices], directed_edges[data_indices], edge_gradient_weights[data_indices], nx.permute(0,2,1))
    return x, y, aux