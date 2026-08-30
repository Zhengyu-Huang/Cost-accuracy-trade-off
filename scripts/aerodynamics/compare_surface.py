"""Exploratory mapping and error diagnostics for OpenFOAM surface outputs.

Run from ``scripts/aerodynamics`` after editing ``openfoam_folder`` and
``test_name`` in the active ``__main__`` block::

    python compare_surface.py

The four input VTK paths must contain a nodal ``p`` array. The active driver
maps the 7000-iteration large-mesh field to the large/2000, medium/1000, and
small/1000 surfaces, writes ``coarse_with_projected_fine_pressure.vtk`` for
each comparison (overwriting the preceding file), and opens two section plots.
This is a single-case exploratory driver, not the six-geometry aggregation.
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt


def read_surface_point_pressure(filename, pressure_name, Cp_name):
    """Extract a clean triangular surface and rename one nodal pressure array.

    Values are copied without nondimensionalization; naming the output ``Cp``
    is only an alias. This is sufficient for relative errors when reference and
    comparison fields use the same pressure scale.
    """
    mesh = pv.read(filename)

    # Normalize both volume and surface inputs to clean triangular PolyData.
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

    # Remove unrelated arrays so sampling cannot collide on fine/coarse names.
    for name in list(surf.point_data.keys()):
        del surf.point_data[name]
    for name in list(surf.cell_data.keys()):
        del surf.cell_data[name]

    surf.point_data[Cp_name] = Cp
    return surf


def plot_section_with_normals(file, plane_origin=(0,0,0), plane_normal=(0,1,0), pressure_name="p"):
    """Visualize a pressure-scaled normal field on a planar surface section."""
    data = read_surface_point_pressure(file, pressure_name, "Cp")
    
    # Compute point normals before slicing so they are interpolated to the cut.
    data_with_normals = data.compute_normals(point_normals=True, cell_normals=False)
    if "Normals" not in data_with_normals.point_data:
        raise RuntimeError("法向计算失败，未生成 'Normals' 数组")
    
    # Slice only after attaching normals to the surface vertices.
    section = data_with_normals.slice(normal=plane_normal, origin=plane_origin).clean()
    
    
    
    # Project the section and its normals to the x-z plotting plane.
    points = section.points
    Cp = section.point_data["Cp"]
    normals = section.point_data["Normals"]
    
    x = points[:, 0]
    z = points[:, 2]
    
    nx_proj = normals[:, 0]
    nz_proj = normals[:, 2]
    
    # Normalize projected normals while guarding against a zero projection.
    magnitude = np.sqrt(nx_proj**2 + nz_proj**2)
    nx_proj_unit = np.divide(nx_proj, magnitude, out=np.zeros_like(nx_proj), where=magnitude>1e-8)
    nz_proj_unit = np.divide(nz_proj, magnitude, out=np.zeros_like(nz_proj), where=magnitude>1e-8)
    
    # Encode the signed pressure value in the normal-arrow length.
    scale = 2.0
    arrow_length = Cp * scale
    u = nx_proj_unit * arrow_length
    v = nz_proj_unit * arrow_length
    
    plt.figure(figsize=(8,6))
    plt.plot(x, z, '.', color = 'black', markersize=0.5, linewidth=0, label='Cross section')
    plt.quiver(x, z, u, v, angles='xy', scale_units='xy', scale=2, alpha=0.7, 
               color='r', width=0.003, label='Cp')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    


def area_weighted_relative_l2_on_coarse(coarse, p_ref, p_cmp, valid):
    """Approximate an area-weighted relative L2 error on a coarse surface.

    ``p_ref`` is the fine reference sampled at coarse vertices and ``p_cmp`` is
    the coarse field. Vertexwise squared values are averaged per triangle and
    integrated using triangle areas; triangles touching invalid vertices are
    excluded.
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




def compute_rel_l2_err(fine_file = "patch_L.vtk", coarse_file = "patch_M.vtk", max_projection_distance=1e-2):
    """Map a fine reference to coarse vertices and report relative errors.

    The published comparison uses the unweighted discrete L1 ratio reported
    here. The area-weighted L2 value is an additional diagnostic, not the metric
    plotted by ``cost_accuracy_trade_off.py``.
    """
    pressure_name = "p"

    fine = read_surface_point_pressure(fine_file, pressure_name, "Cp_fine")
    coarse = read_surface_point_pressure(coarse_file, pressure_name, "Cp_coarse")

    # Project each coarse vertex to its nearest point on the fine surface.
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
    
    
    # Combine PyVista's sampling mask with the optional geometric tolerance.
    valid = sample_valid.copy()

    if max_projection_distance is not None:
        valid = valid & (projection_distance <= max_projection_distance)

    print("\n采样检查:")
    print("sample valid points:", int(sample_valid.sum()))
    print(f"sample valid ratio : {100.0 * sample_valid.mean():.2f}%")
    print("final valid points :", int(valid.sum()))
    print(f"final valid ratio  : {100.0 * valid.mean():.2f}%")
    
    
    # Assemble reference and difference arrays on the coarse vertices.
    p_coarse = np.asarray(coarse.point_data["Cp_coarse"]).astype(float)

    p_fine_proj = np.full_like(p_coarse, np.nan, dtype=float)
    p_fine_proj[valid] = p_fine_sampled[valid]

    p_diff = np.full_like(p_coarse, np.nan, dtype=float)
    p_abs_diff = np.full_like(p_coarse, np.nan, dtype=float)
    

    p_diff[valid] = p_fine_proj[valid] - p_coarse[valid]
    p_abs_diff[valid] = np.abs(p_diff[valid])

    # These are unweighted nodal ratios; mesh point density affects the metric.
    rel_l2_error = np.linalg.norm(p_diff[valid], ord=2) / np.linalg.norm(p_fine_proj[valid], ord=2)
    print("relative L2 error is ", rel_l2_error)
    rel_l1_error = np.linalg.norm(p_diff[valid], ord=1) / np.linalg.norm(p_fine_proj[valid], ord=1)
    print("relative L1 error is ", rel_l1_error)
    print(np.linalg.norm(p_diff[valid]), np.linalg.norm(p_fine_proj[valid]))
    

    
    # Also report an area-weighted L2 diagnostic on valid coarse triangles.
    rel_l2_area = area_weighted_relative_l2_on_coarse(
        coarse=coarse,
        p_ref=p_fine_proj,
        p_cmp=p_coarse,
        valid=valid,
    )
    print("area weighted relative L2 error is ", rel_l2_area)
    
    
    # Save mapped fields on the coarse mesh for visual inspection.
    out = coarse.copy()
    out.point_data["Cp_coarse"] = p_coarse
    out.point_data["Cp_fine_projected_on_coarse"] = p_fine_proj
    out.point_data["Cp_diff_coarse_minus_fine"] = p_diff
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
