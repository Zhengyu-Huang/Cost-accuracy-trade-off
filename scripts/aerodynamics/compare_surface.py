import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt


def read_surface_point_pressure(filename, pressure_name, Cp_name):
    mesh = pv.read(filename)

    # 如果是体网格，提取表面；如果本来就是表面网格，也可以这样处理
    surf = mesh.extract_surface().triangulate().clean()

    print(f"\n读取文件: {filename}")
    print("Point data arrays:", list(surf.point_data.keys()))
    print("Cell data arrays: ", list(surf.cell_data.keys()))

    if pressure_name not in surf.point_data:
        raise KeyError(
            f"没有在 point_data 里找到 '{pressure_name}'。\n"
            f"现有 point_data: {list(surf.point_data.keys())}\n"
            f"现有 cell_data : {list(surf.cell_data.keys())}\n"
            "如果你的 pressure 是 cell data，请用下面的 cell data 版本。"
        )

    Cp = np.asarray(surf.point_data[pressure_name]).astype(float).copy() # /(0.5*30*30)

    # 清掉原来的数组，避免 fine/coarse 同名变量冲突
    for name in list(surf.point_data.keys()):
        del surf.point_data[name]
    for name in list(surf.cell_data.keys()):
        del surf.cell_data[name]

    surf.point_data[Cp_name] = Cp
    return surf


def plot_section_with_normals(file, plane_origin=(0,0,0), plane_normal=(0,1,0), pressure_name="p"):
    # 1. 读取表面数据，压力以点数据形式存在
    data = read_surface_point_pressure(file, pressure_name, "Cp")
    
    # 2. 【关键修改】先计算原始表面网格的法向，存储到点数据中
    #    使用点法向（point_normals=True），并将结果命名为 "Normals"
    data_with_normals = data.compute_normals(point_normals=True, cell_normals=False)
    # 检查 Normals 是否存在
    if "Normals" not in data_with_normals.point_data:
        raise RuntimeError("法向计算失败，未生成 'Normals' 数组")
    
    # 3. 现在对带有法向信息的网格进行切片
    section = data_with_normals.slice(normal=plane_normal, origin=plane_origin).clean()
    
    
    
    # 提取坐标、压力和法向
    points = section.points
    Cp = section.point_data["Cp"]
    normals = section.point_data["Normals"]
    
    # 4. 投影到 x-z 平面（忽略 y 分量）
    x = points[:, 0]
    z = points[:, 2]
    
    # 法向投影到 x-z 平面 (只取 x 和 z 分量)
    nx_proj = normals[:, 0]
    nz_proj = normals[:, 2]
    
    # 单位化投影后的法向（避免长度失真）
    magnitude = np.sqrt(nx_proj**2 + nz_proj**2)
    # 避免除以零
    nx_proj_unit = np.divide(nx_proj, magnitude, out=np.zeros_like(nx_proj), where=magnitude>1e-8)
    nz_proj_unit = np.divide(nz_proj, magnitude, out=np.zeros_like(nz_proj), where=magnitude>1e-8)
    
    # 5. 箭头长度 = 压力值（可乘以一个全局缩放系数，例如 0.1）
    scale = 2.0   # 根据需要调整
    arrow_length = Cp * scale
    u = nx_proj_unit * arrow_length
    v = nz_proj_unit * arrow_length
    
    # 6. 绘制
    plt.figure(figsize=(8,6))
    # 画出曲线
    plt.plot(x, z, '.', color = 'black', markersize=0.5, linewidth=0, label='Cross section')
    # 画出法向箭头
    plt.quiver(x, z, u, v, angles='xy', scale_units='xy', scale=2, alpha=0.7, 
               color='r', width=0.003, label='Cp')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    


# def compute_rel_l2_err(fine_file = "patch_L.vtk", coarse_file = "patch_M.vtk", max_projection_distance=1e-2):
#     # 改成你 vtk 文件里的压力变量名
#     pressure_name = "p"

#     fine = read_surface_point_pressure(fine_file, pressure_name, "Cp_fine")
#     coarse = read_surface_point_pressure(coarse_file, pressure_name, "Cp_coarse")

#     # closest_point
#     closest_cell_ids, proj_pts = coarse.find_closest_cell(
#         fine.points,
#         return_closest_point=True,
#     )

#     closest_cell_ids = np.asarray(closest_cell_ids).astype(np.int64)
#     proj_pts = np.asarray(proj_pts).astype(float)

#     projection_distance = np.linalg.norm(
#         fine.points - proj_pts,
#         axis=1,
#     )

#     print("\n投影检查:")
#     print("fine points:", fine.n_points)
#     print("projected points:", proj_pts.shape[0])
#     print(f"max projection distance : {np.nanmax(projection_distance):.6e}")
#     print(f"mean projection distance: {np.nanmean(projection_distance):.6e}")

#     sample_points = pv.PolyData(proj_pts)
#     sampled_data = sample_points.sample(coarse)

#     sample_valid = np.asarray(
#         sampled_data.point_data["vtkValidPointMask"]
#     ).astype(bool)

#     if "Cp_coarse" not in sampled_data.point_data:
#         raise KeyError(
#             "sample 后没有找到 p_coarse。\n"
#             f"sampled point_data: {list(sampled_data.point_data.keys())}"
#         )

#     p_coarse_sampled = np.asarray(
#         sampled_data.point_data["Cp_coarse"]
#     ).astype(float)
    
    
#     # 如果设置了最大投影距离，则进一步过滤
#     valid = sample_valid.copy()

#     if max_projection_distance is not None:
#         valid = valid & (projection_distance <= max_projection_distance)

#     print("\n采样检查:")
#     print("sample valid points:", int(sample_valid.sum()))
#     print(f"sample valid ratio : {100.0 * sample_valid.mean():.2f}%")
#     print("final valid points :", int(valid.sum()))
#     print(f"final valid ratio  : {100.0 * valid.mean():.2f}%")
    
    
#     # ============================================================
#     # 3. 计算误差，并保存到原始 fine mesh 上
#     # ============================================================
#     p_fine = np.asarray(fine.point_data["Cp_fine"]).astype(float)

#     p_coarse_proj = np.full_like(p_fine, np.nan, dtype=float)
#     p_coarse_proj[valid] = p_coarse_sampled[valid]

#     p_diff = np.full_like(p_fine, np.nan, dtype=float)
#     p_abs_diff = np.full_like(p_fine, np.nan, dtype=float)
#     p_rel_error = np.full_like(p_fine, np.nan, dtype=float)

#     p_diff[valid] = p_coarse_proj[valid] - p_fine[valid]
#     p_abs_diff[valid] = np.abs(p_diff[valid])
#     rel_l2_error = np.linalg.norm(p_diff[valid]) / np.linalg.norm(p_fine[valid])
#     print("relative L2 error is ", rel_l2_error)


def area_weighted_relative_l2_on_coarse(coarse, p_ref, p_cmp, valid):
    """
    在 coarse surface 上计算面积加权 relative L2 error:

        sqrt( integral((p_cmp - p_ref)^2 dA) / integral(p_ref^2 dA) )

    其中：
        p_ref = fine projected onto coarse
        p_cmp = coarse
    """

    faces = np.asarray(coarse.faces).reshape(-1, 4)

    if not np.all(faces[:, 0] == 3):
        raise RuntimeError("coarse mesh 不是纯三角形，请确认 triangulate() 是否成功。")

    tri = faces[:, 1:4]

    pts = coarse.points
    a = pts[tri[:, 0]]
    b = pts[tri[:, 1]]
    c = pts[tri[:, 2]]

    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    tri_valid = valid[tri].all(axis=1)

    err2_point = (p_cmp - p_ref) ** 2
    ref2_point = p_ref ** 2

    err2_cell = np.mean(err2_point[tri], axis=1)
    ref2_cell = np.mean(ref2_point[tri], axis=1)

    mask = (
        tri_valid
        & np.isfinite(err2_cell)
        & np.isfinite(ref2_cell)
        & np.isfinite(area)
        & (area > 0.0)
    )
    
    numerator = np.sum(area[mask] * err2_cell[mask])
    denominator = np.sum(area[mask] * ref2_cell[mask])

    if denominator <= 0.0:
        return np.nan

    return np.sqrt(numerator / denominator)




# Compare on coarse grid 
def compute_rel_l2_err(fine_file = "patch_L.vtk", coarse_file = "patch_M.vtk", max_projection_distance=1e-2):
    # 改成你 vtk 文件里的压力变量名
    pressure_name = "p"

    fine = read_surface_point_pressure(fine_file, pressure_name, "Cp_fine")
    coarse = read_surface_point_pressure(coarse_file, pressure_name, "Cp_coarse")

    # closest_point
    closest_cell_ids, proj_pts = fine.find_closest_cell(
        coarse.points,
        return_closest_point=True,
    )

    closest_cell_ids = np.asarray(closest_cell_ids).astype(np.int64)
    proj_pts = np.asarray(proj_pts).astype(float)

    projection_distance = np.linalg.norm(
        coarse.points - proj_pts,
        axis=1,
    )

    print("\n投影检查:")
    print("fine points:", coarse.n_points)
    print("projected points:", proj_pts.shape[0])
    print(f"max projection distance : {np.nanmax(projection_distance):.6e}")
    print(f"mean projection distance: {np.nanmean(projection_distance):.6e}")

    sample_points = pv.PolyData(proj_pts)
    sampled_data = sample_points.sample(fine)

    sample_valid = np.asarray(
        sampled_data.point_data["vtkValidPointMask"]
    ).astype(bool)

    if "Cp_fine" not in sampled_data.point_data:
        raise KeyError(
            "sample 后没有找到 p_fine。\n"
            f"sampled point_data: {list(sampled_data.point_data.keys())}"
        )

    p_fine_sampled = np.asarray(
        sampled_data.point_data["Cp_fine"]
    ).astype(float)
    
    
    # 如果设置了最大投影距离，则进一步过滤
    valid = sample_valid.copy()

    if max_projection_distance is not None:
        valid = valid & (projection_distance <= max_projection_distance)

    print("\n采样检查:")
    print("sample valid points:", int(sample_valid.sum()))
    print(f"sample valid ratio : {100.0 * sample_valid.mean():.2f}%")
    print("final valid points :", int(valid.sum()))
    print(f"final valid ratio  : {100.0 * valid.mean():.2f}%")
    
    
    # ============================================================
    # 3. 计算误差，并保存到原始 fine mesh 上
    # ============================================================
    p_coarse = np.asarray(coarse.point_data["Cp_coarse"]).astype(float)

    p_fine_proj = np.full_like(p_coarse, np.nan, dtype=float)
    p_fine_proj[valid] = p_fine_sampled[valid]

    p_diff = np.full_like(p_coarse, np.nan, dtype=float)
    p_abs_diff = np.full_like(p_coarse, np.nan, dtype=float)
    

    p_diff[valid] = p_fine_proj[valid] - p_coarse[valid]
    p_abs_diff[valid] = np.abs(p_diff[valid])

    rel_l2_error = np.linalg.norm(p_diff[valid], ord=2) / np.linalg.norm(p_fine_proj[valid], ord=2)
    print("relative L2 error is ", rel_l2_error)
    rel_l1_error = np.linalg.norm(p_diff[valid], ord=1) / np.linalg.norm(p_fine_proj[valid], ord=1)
    print("relative L1 error is ", rel_l1_error)
    print(np.linalg.norm(p_diff[valid]), np.linalg.norm(p_fine_proj[valid]))
    

    
    # 面积加权 relative L2 error，更推荐用于 surface pressure
    rel_l2_area = area_weighted_relative_l2_on_coarse(
        coarse=coarse,
        p_ref=p_fine_proj,
        p_cmp=p_coarse,
        valid=valid,
    )
    print("area weighted relative L2 error is ", rel_l2_area)
    
    
    # save data for visualization
    # 输出 coarse mesh
    out = coarse.copy()
    out.point_data["Cp_coarse"] = p_coarse
    out.point_data["Cp_fine_projected_on_coarse"] = p_fine_proj
    out.point_data["Cp_diff_coarse_minus_fine"] = p_diff
    # 推荐保存 vtp；也可以同时保存 legacy vtk
    out.save("coarse_with_projected_fine_pressure.vtk", binary=True)
    print("saved:")
    print("coarse_with_projected_fine_pressure.vtk")
    
if __name__ == "__main__":
    openfoam_folder="/lustre/home/2306192137/OpenFOAM/"
    test_name = "E_S_WW_WM_001" #"drivaerFastback"
    large_mesh = openfoam_folder + test_name + "_L/postProcessing/car/7000/patch.vtk"
    large_short_mesh = openfoam_folder + test_name + "_L/postProcessing/car/2000/patch.vtk"
    middle_mesh = openfoam_folder + test_name + "_M/postProcessing/car/1000/patch.vtk"
    small_mesh = openfoam_folder + test_name + "_S/postProcessing/car/1000/patch.vtk"
    max_projection_distance = 1e-2
    print("Compare large meshes with different number of steps")
    compute_rel_l2_err(fine_file = large_mesh, coarse_file = large_short_mesh, max_projection_distance=max_projection_distance)
    print("Compare large and middle meshes")
    compute_rel_l2_err(fine_file = large_mesh, coarse_file = middle_mesh, max_projection_distance=max_projection_distance)
    print("Compare large and small meshes")
    compute_rel_l2_err(fine_file = large_mesh, coarse_file = small_mesh, max_projection_distance=max_projection_distance)
    
    plot_section_with_normals(large_mesh)
    plot_section_with_normals(middle_mesh)
    plt.show()
