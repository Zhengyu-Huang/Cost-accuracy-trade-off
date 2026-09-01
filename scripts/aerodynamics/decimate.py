#!/usr/bin/env python3
"""Decimate DrivAerNet++ VTK surfaces while preserving pressure data.

The workflow converts VTK geometry to PLY for PyMeshLab, performs quadric
edge-collapse decimation, transfers the original nodal pressure by nearest
neighbor, and writes a VTK surface with consistently oriented normals.

Run either mode from ``scripts/aerodynamics``::

    python decimate.py INPUT.vtk OUTPUT.vtk 20000
    python decimate.py INPUT_VTK_DIR INPUT_PLY_DIR OUTPUT_PLY_DIR OUTPUT_VTK_DIR 10000

Inputs must be triangular legacy VTK PolyData with a nodal array named ``p``.
Single-file mode writes ``OUTPUT.vtk`` and retains adjacent source/decimated
PLY files. Directory mode writes one PLY/VTK pair per source file. ``__main__``
selects single-file mode for 3--4 command-line arguments and directory mode for
5--6; in practice, include the directory-mode target count as shown.
"""

import sys
import os
import tempfile
import pymeshlab
import numpy as np
import vtk
from scipy.spatial import cKDTree


def convert_vtk_to_ply(vtk_file, ply_file):
    """Convert VTK PolyData geometry to PLY; point fields are not retained."""
    print(f"正在读取 VTK 文件: {vtk_file}")
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file)
    reader.Update()

    polydata = reader.GetOutput()
    if polydata.GetNumberOfPoints() == 0:
        raise Exception(f"VTK 文件 '{vtk_file}' 无有效点数据。")

    print(f"VTK 读取成功: {polydata.GetNumberOfPoints()} 顶点, {polydata.GetNumberOfPolys()} 面")
    print(f"正在转换为 PLY 格式: {ply_file}")
    writer = vtk.vtkPLYWriter()
    writer.SetFileName(ply_file)
    writer.SetInputData(polydata)
    writer.Write()
    print("转换完成。")

def simplify_mesh(input_ply_file, output_ply_file, target_vertices):
    """Decimate a PLY surface to approximately ``target_vertices`` vertices.

    ``optimalplacement=False`` restricts edge collapses to existing vertex
    positions, which makes nearest-neighbor transfer from the source mesh
    well-defined and avoids introducing interpolated coordinates.
    """
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_ply_file)
    print(f"加载 PLY 成功: {ms.current_mesh().vertex_number()} 顶点")


    print(f"正在简化至约 {target_vertices} 顶点...")
    
    # PyMeshLab accepts a target fraction rather than a target vertex count.
    targetperc = target_vertices / ms.current_mesh().vertex_number()
    ms.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetperc=targetperc,
        preservenormal=True,
        preserveboundary=True,
        preservetopology=True,
        planarquadric=True,
        optimalplacement=False,
    )

    final_vertices = ms.current_mesh().vertex_number()
    print(f"简化完成: {final_vertices} 顶点")

    # Remove connectivity defects introduced or exposed by decimation.
    ms.apply_filter('meshing_remove_duplicate_faces')
    ms.apply_filter('meshing_remove_duplicate_vertices')

    ms.apply_filter('meshing_remove_unreferenced_vertices')
    ms.apply_filter('meshing_remove_null_faces')

    

    final_vertices = ms.current_mesh().vertex_number()
    print(f"清理完成: {final_vertices} 顶点")
    ms.save_current_mesh(output_ply_file)
    print(f"结果已保存至: {output_ply_file}")


    




def transfer_scalars_vtk_to_ply(input_vtk_file, ply_mesh_file, output_vtk_file):
    """Transfer nodal ``p`` to a decimated mesh and orient its normals.

    Each retained PLY vertex receives the value at its nearest source VTK
    vertex. The nearest-neighbor distances printed below diagnose whether the
    decimated geometry still coincides with the source surface.
    """
    # Read source coordinates and the nodal pressure field.
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(input_vtk_file)
    reader.Update()
    original_mesh = reader.GetOutput()
    
    orig_points = original_mesh.GetPoints()
    orig_scalars = original_mesh.GetPointData().GetArray("p")
    
    if orig_scalars is None:
        raise ValueError("原始VTK文件中没有点标量数据！")
    
    num_orig_points = orig_points.GetNumberOfPoints()
    orig_coords = np.array([orig_points.GetPoint(i) for i in range(num_orig_points)])
    orig_p_values = np.array([orig_scalars.GetValue(i) for i in range(num_orig_points)])
    
    print(f"原始模型: {num_orig_points} 个顶点，标量范围 [{orig_p_values.min():.3f}, {orig_p_values.max():.3f}]")
    
    # Read the decimated PLY geometry, which contains no pressure field.
    ply_reader = vtk.vtkPLYReader()
    ply_reader.SetFileName(ply_mesh_file)
    ply_reader.Update()
    ply_mesh = ply_reader.GetOutput()
    
    ply_points = ply_mesh.GetPoints()
    num_ply_points = ply_points.GetNumberOfPoints()
    ply_coords = np.array([ply_points.GetPoint(i) for i in range(num_ply_points)])
    
    print(f"简化模型: {num_ply_points} 个顶点")
    
    # Map every decimated vertex to its nearest vertex on the source mesh.
    tree = cKDTree(orig_coords)
    distances, indices = tree.query(ply_coords, k=1)
    
    # Attach the mapped values under the original VTK array name.
    new_scalars = vtk.vtkDoubleArray()
    new_scalars.SetName("p")
    new_scalars.SetNumberOfValues(num_ply_points)
    
    for i, idx in enumerate(indices):
        new_scalars.SetValue(i, orig_p_values[idx])
    
    ply_mesh.GetPointData().SetScalars(new_scalars)


    print("标量值后网格格点个数: ", ply_mesh.GetPoints().GetNumberOfPoints())

    print("正在统一法线方向，确保所有三角形法线向外...")
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(ply_mesh)
    normals_filter.SetConsistency(True)           # Keep neighboring face orientations consistent.
    normals_filter.SetAutoOrientNormals(True)     # Infer an outward orientation for closed components.
    normals_filter.SetNonManifoldTraversal(True)  # Traverse non-manifold edges when propagating orientation.
    normals_filter.SetSplitting(False)            # Preserve one value per existing vertex.
    normals_filter.SetFlipNormals(False)
    normals_filter.Update()
    ply_mesh = normals_filter.GetOutput()
    # vtkPolyDataNormals passes the mapped point-data array through unchanged.
    print("法线统一完成")


    print("最后网格格点个数: ", ply_mesh.GetPoints().GetNumberOfPoints())
    
    # Write geometry, mapped pressure, and oriented normals to legacy VTK.
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_vtk_file)
    writer.SetInputData(ply_mesh)
    writer.Write()
    
    print(f"已保存带有标量 p 的 VTK 文件: {output_vtk_file}")
    print(f"最近邻距离统计: 最大距离 = {distances.max():.6f}, 平均距离 = {distances.mean():.6f}")




def preprocess():
    """Run the single-file VTK-to-decimated-VTK workflow.

    Intermediate source and decimated PLY files are retained beside the input
    and output paths.
    """
    if len(sys.argv) < 3:
        print("用法: python decimate.py 输入.vtk 输出.vtk [目标顶点数]")
        print("示例: python decimate.py E_S_WWC_WM_018.vtk E_S_WWC_WM_018_decimate.ply 30000")
        sys.exit(1)

    input_vtk_file = sys.argv[1]
    output_vtk_file = sys.argv[2]
    target = int(sys.argv[3]) if len(sys.argv) > 3 else 30000

    if not os.path.exists(input_vtk_file):
        print(f"错误: 文件不存在 '{input_vtk_file}'")
        sys.exit(1)

    # Use neighboring PLY paths as the intermediate conversion files.
    input_base = os.path.splitext(input_vtk_file)[0]
    output_base = os.path.splitext(output_vtk_file)[0]

    input_ply_file = input_base + '.ply'
    output_ply_file = output_base + '.ply'
    convert_vtk_to_ply(input_vtk_file, input_ply_file)

    simplify_mesh(input_ply_file, output_ply_file, target)
    

    transfer_scalars_vtk_to_ply(input_vtk_file, output_ply_file, output_vtk_file)




def drivaernet_preprocess():
    """Apply the decimation workflow to every VTK file in one category.

    The four directory arguments hold source VTK, source PLY, decimated PLY,
    and decimated VTK files, respectively.
    """
    if len(sys.argv) < 5:
        print("用法: ython decimate.py 输入文件夹名 输入PLY文件夹名  输出PLY文件夹名 输出文件夹名 [目标顶点数]")
        sys.exit(1)

    input_vtk_folder = sys.argv[1]
    input_ply_folder = sys.argv[2]
    output_ply_folder = sys.argv[3]
    output_vtk_folder = sys.argv[4]

    os.makedirs(input_ply_folder, exist_ok=True)
    os.makedirs(output_ply_folder, exist_ok=True)
    os.makedirs(output_vtk_folder, exist_ok=True)

    target = int(sys.argv[5]) if len(sys.argv) > 4 else 30000

    for dirpath, _, filenames in os.walk(input_vtk_folder):
        for filename in filenames:
            if filename.endswith(".vtk"):
                base = os.path.splitext(filename)[0]
                
                ply_file = base + '.ply'
                vtk_file = base + '.vtk'

                input_vtk_file, input_ply_file = os.path.join(input_vtk_folder, vtk_file), os.path.join(input_ply_folder, ply_file)
                output_ply_file, output_vtk_file = os.path.join(output_ply_folder, ply_file), os.path.join(output_vtk_folder, vtk_file)
                
                convert_vtk_to_ply(input_vtk_file, input_ply_file)

                simplify_mesh(input_ply_file, output_ply_file, target)
                
                transfer_scalars_vtk_to_ply(input_vtk_file, output_ply_file, output_vtk_file)


if __name__ == "__main__":
    argc = len(sys.argv)

    if argc == 5 or argc == 6:
        drivaernet_preprocess()
    
    if argc == 3 or argc == 4:
        preprocess()
        

    # python decimate.py "/lustre/home/2306192137/Cost-accuracy-trade-off/data/aerodynamics/PressureVTK/E_S_WWC_WM"                   \
    #                    "/lustre/home/2306192137/Cost-accuracy-trade-off/data/aerodynamics/PressurePLY/E_S_WWC_WM"                   \
    #                    "/lustre/home/2306192137/Cost-accuracy-trade-off/data/aerodynamics/PressurePLY_Processed/E_S_WWC_WM"         \
    #                    "/lustre/home/2306192137/Cost-accuracy-trade-off/data/aerodynamics/test" 40000 
    # main()
    # input_vtk_file, ply_mesh_file, output_vtk_file = "E_S_WWC_WM_018.vtk", "E_S_WWC_WM_018_decimate.ply", "E_S_WWC_WM_018_decimate.vtk"
    # transfer_scalars_vtk_to_ply(input_vtk_file, ply_mesh_file, output_vtk_file)
