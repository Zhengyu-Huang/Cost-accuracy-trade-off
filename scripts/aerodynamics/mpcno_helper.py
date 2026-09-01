"""Shared data-loading and tensor-assembly helpers for the M-PCNO workflow.

This module is imported by preprocessing, training, evaluation, and plotting
scripts and has no ``__main__`` entry point. It expects decimated triangular
VTK surfaces with nodal ``p`` under the 15 category directories listed below.
It does not write files directly; callers own archive and checkpoint output.
"""

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
    """Append one triangular VTK surface and its nodal features to lists.

    Each feature row contains the three point-normal components followed by
    the pressure coefficient. Connectivity is stored in the mesh preprocessor's
    ``[element_dimension, vertex_ids...]`` convention.
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

    # Legacy VTK triangles are flattened as [3, i, j, k, 3, ...].
    polys = polydata.GetPolys()
    cell_array = vtk_to_numpy(polys.GetData())
    elems = cell_array.reshape(-1, 4)
    elems[:,0] = 2  # Replace the vertex count with the surface dimension.


    # pressure
    scalars = polydata.GetPointData().GetArray("p")
    pressure_feature = np.array([scalars.GetValue(i) for i in range(num_points)])
    # The exported ``p`` is treated as kinematic gauge pressure, so rho=1 and
    # p_inf=0 are implicit in Cp = p/(0.5*U_inf**2).
    u_inf = 30
    cp_feature = pressure_feature / (0.5*u_inf**2)

    # Compute smooth point normals without splitting vertices at sharp edges.
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(reader.GetOutput())
    normals.SetComputePointNormals(True)
    normals.SetComputeCellNormals(False)
    normals.SetSplitting(False)
    normals.Update()
    point_normals = vtk_to_numpy(normals.GetOutput().GetPointData().GetNormals())
    
    elem_features = np.column_stack((point_normals, cp_feature))  
    
    nodes_list.append(nodes)
    elems_list.append(elems)
    elem_features_list.append(elem_features)


    # Optional round-trip check for geometry, pressure, and normals.
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
    """Load up to ``n_each`` surfaces from every configured category.

    Files are consumed in the order returned by ``os.walk``. Consequently,
    the selected geometries depend on filesystem ordering unless callers sort
    the file lists before regenerating the archives.
    """
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
    """Select and reorder a seeded training/test subset.

    The first ``n_train`` shuffled indices form the training block and the last
    ``n_test`` form the test block. Samples between those two blocks are
    discarded when the source archive is larger than the requested subset.
    """
    np.random.seed(seed)
    
    ndata = data["nodes"].shape[0]
    assert(ndata >= n_train + n_test)
    print("Total data number =", ndata, " n_train = ", n_train, " n_test = ", n_test)
    random_indices = np.arange(ndata)
    np.random.shuffle(random_indices)
    
    # Keep training samples first because downstream code slices by position.
    train_indices = random_indices[:n_train]
    test_indices = random_indices[-n_test:]
    indices = np.concatenate([train_indices, test_indices])
    
    
    data = {key: value[indices] for key, value in data.items()}
    names_array = names_array[indices]
    
    # Report category counts after reordering as a split sanity check.
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


def gen_data_tensors(data_indices, nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights, f_in_dim, f_out_dim):
    """Assemble model inputs, targets, and geometry tensors for selected cases.

    For this benchmark ``features`` stores point normals followed by ``Cp``.
    Inputs concatenate the requested physical features, normals, and Cartesian
    coordinates; ``aux`` preserves the order expected by ``MPCNO.forward``.
    """
    nodes_input = nodes.clone()
    ndim = nodes.shape[-1]
    # Input channels are outward normals followed by Cartesian coordinates.
    x = torch.cat((features[data_indices][...,:f_in_dim+ndim], nodes_input[data_indices, ...]), -1)
    # The final feature channel is the scalar Cp target.
    y = features[data_indices][...,-f_out_dim:]
    # outward normal
    nx = features[data_indices][...,f_in_dim:f_in_dim+ndim]
    aux = (node_mask[data_indices], nodes[data_indices], node_weights[data_indices], directed_edges[data_indices], edge_gradient_weights[data_indices], nx.permute(0,2,1))
    return x, y, aux
