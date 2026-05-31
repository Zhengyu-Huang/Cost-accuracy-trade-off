#!/usr/bin/env python3
"""
VTK 网格自动化简化工具：
1. 使用 vtk 库读取 .vtk 文件，并转换为 PyMeshLab 支持的 .ply 格式。
2. 对转换后的网格执行二次误差边折叠简化，保留原始顶点坐标。
"""

import sys
import os
import tempfile
import pymeshlab
import numpy as np
# 需要导入 vtk
import vtk
from scipy.spatial import cKDTree


def convert_vtk_to_ply(vtk_file, ply_file):
    """
    使用 vtk.vtkPolyDataReader 读取 .vtk 文件，并保存为 .ply 格式, 只保存了网格信息
    """
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
    """
    对 PLY 网格执行简化操作，关键参数 'optimalposition=False' 确保只使用原始顶点坐标。
    """
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_ply_file)
    print(f"加载 PLY 成功: {ms.current_mesh().vertex_number()} 顶点")


    print(f"正在简化至约 {target_vertices} 顶点...")
    
    # 简化网格
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

    # 清理网格
    ms.apply_filter('meshing_remove_duplicate_faces')
    ms.apply_filter('meshing_remove_duplicate_vertices')

    ms.apply_filter('meshing_remove_unreferenced_vertices')  # 移除未被任何面引用的孤立顶点[reference:0]
    ms.apply_filter('meshing_remove_null_faces')             # 删除面积为0的退化面[reference:1]

    

    final_vertices = ms.current_mesh().vertex_number()
    print(f"清理完成: {final_vertices} 顶点")
    ms.save_current_mesh(output_ply_file)
    print(f"结果已保存至: {output_ply_file}")


    




def transfer_scalars_vtk_to_ply(input_vtk_file, ply_mesh_file, output_vtk_file):
    # 1. 读取原始 VTK，获取点坐标和标量 p
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
    
    # 2. 读取简化后的 PLY 文件
    ply_reader = vtk.vtkPLYReader()
    ply_reader.SetFileName(ply_mesh_file)
    ply_reader.Update()
    ply_mesh = ply_reader.GetOutput()
    
    ply_points = ply_mesh.GetPoints()
    num_ply_points = ply_points.GetNumberOfPoints()
    ply_coords = np.array([ply_points.GetPoint(i) for i in range(num_ply_points)])
    
    print(f"简化模型: {num_ply_points} 个顶点")
    
    # 3. 建立原始点云的 KDTree，查询每个 PLY 点的最近邻
    tree = cKDTree(orig_coords)
    distances, indices = tree.query(ply_coords, k=1)
    
    # 4. 将对应的标量值赋给 PLY 的点数据
    new_scalars = vtk.vtkDoubleArray()
    new_scalars.SetName("p")   # 保持原来的标量名
    new_scalars.SetNumberOfValues(num_ply_points)
    
    for i, idx in enumerate(indices):
        new_scalars.SetValue(i, orig_p_values[idx])
    
    ply_mesh.GetPointData().SetScalars(new_scalars)



    print("正在统一法线方向，确保所有三角形法线向外...")
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(ply_mesh)
    normals_filter.SetConsistency(True)           # 使相邻面的法线方向一致
    normals_filter.SetAutoOrientNormals(True)     # 自动将法线翻转向外（适用于封闭网格）
    normals_filter.SetNonManifoldTraversal(True)  # 处理非流形边，增强对开放网格的鲁棒性
    normals_filter.SetFlipNormals(False)          # 不额外翻转
    normals_filter.Update()
    ply_mesh = normals_filter.GetOutput()         # 替换为法线统一后的网格
    # 注：vtkPolyDataNormals 会自动传递原有的点标量数组，无需额外操作
    print("法线统一完成")



    
    # 5. 保存为新的 VTK 文件
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_vtk_file)
    writer.SetInputData(ply_mesh)
    writer.Write()
    
    print(f"已保存带有标量 p 的 VTK 文件: {output_vtk_file}")
    print(f"最近邻距离统计: 最大距离 = {distances.max():.6f}, 平均距离 = {distances.mean():.6f}")




def preprocess():
    """
    把 vtk 文件简化 成大概目标顶点数的三角形网格，保持法向朝外
    用法: python decimate.py 输入.vtk 输出.vtk [目标顶点数]
    会储存中间文件: 输入.ply 输出.ply
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

    # 替换后缀为 .ply
    input_base = os.path.splitext(input_vtk_file)[0]
    output_base = os.path.splitext(output_vtk_file)[0]

    input_ply_file = input_base + '.ply'
    output_ply_file = output_base + '.ply'
    convert_vtk_to_ply(input_vtk_file, input_ply_file)

    simplify_mesh(input_ply_file, output_ply_file, target)
    

    transfer_scalars_vtk_to_ply(input_vtk_file, output_ply_file, output_vtk_file)




def drivaernet_preprocess():
    """
    DrivAerNet++ dataset preprocess
    把 vtk 文件简化 成大概目标顶点数的三角形网格，保持法向朝外
    用法: python decimate.py 输入文件夹名 输入PLY文件夹名  输出PLY文件夹名 输出文件夹名 [目标顶点数]
    会储存中间文件: 输入.ply 输出.ply
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
    drivaernet_preprocess()

    # python decimate.py "/lustre/home/2306192137/Cost-accuracy-trade-off/data/aerodynamics/PressureVTK/E_S_WWC_WM"                   \
    #                    "/lustre/home/2306192137/Cost-accuracy-trade-off/data/aerodynamics/PressurePLY/E_S_WWC_WM"                   \
    #                    "/lustre/home/2306192137/Cost-accuracy-trade-off/data/aerodynamics/PressurePLY_Processed/E_S_WWC_WM"         \
    #                    "/lustre/home/2306192137/Cost-accuracy-trade-off/data/aerodynamics/PressureVTK_Processed/E_S_WWC_WM" 40000 
    # main()
    # input_vtk_file, ply_mesh_file, output_vtk_file = "E_S_WWC_WM_018.vtk", "E_S_WWC_WM_018_decimate.ply", "E_S_WWC_WM_018_decimate.vtk"
    # transfer_scalars_vtk_to_ply(input_vtk_file, ply_mesh_file, output_vtk_file)
