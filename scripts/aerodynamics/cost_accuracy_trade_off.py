import numpy as np
import pyvista as pv


def _as_tri_surface_with_point_field(mesh, field_name="Cp"):
    """
    Prepare source/coarse mesh.

    The source mesh must have field_name as point data for linear interpolation.
    If field_name is only cell data, convert it to point data by averaging.
    """
    mesh = mesh.extract_surface()

    if field_name not in mesh.point_data:
        if field_name in mesh.cell_data:
            mesh = mesh.cell_data_to_point_data(pass_cell_data=True)
        else:
            raise KeyError(
                f"Cannot find '{field_name}' in point_data or cell_data. "
                f"Available arrays: {mesh.array_names}"
            )

    # Triangulate so every element has 3 vertices.
    mesh = mesh.triangulate()

    if field_name not in mesh.point_data:
        raise RuntimeError(f"'{field_name}' is not available as point data after conversion.")

    return mesh


def _as_tri_surface_with_cell_field(mesh, field_name="Cp"):
    """
    Prepare target/fine mesh.

    The target mesh will be used with cell centers and cell areas.
    The fine Cp field is converted to cell data if needed.
    """
    mesh = mesh.extract_surface().triangulate()

    if field_name not in mesh.cell_data:
        if field_name in mesh.point_data:
            mesh = mesh.point_data_to_cell_data(pass_point_data=True)
        else:
            raise KeyError(
                f"Cannot find '{field_name}' in point_data or cell_data. "
                f"Available arrays: {mesh.array_names}"
            )

    return mesh


def _triangle_barycentric_batch(p, a, b, c, eps=1.0e-14):
    """
    Compute barycentric coordinates of points p with respect to triangles (a,b,c).

    Parameters
    ----------
    p, a, b, c : ndarray
        Shape (M, 3)

    Returns
    -------
    lambdas : ndarray
        Shape (M, 3), barycentric coordinates.
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)

    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < eps, np.nan, denom)

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return np.stack([u, v, w], axis=1)


def project_coarse_cp_to_fine_faces(
    fine_file,
    coarse_file,
    output_file="Cp_error_projected.vtk",
    field_name="Cp",
    distance_tol=None,
    normal_dot_tol=0.7,
):
    """
    Project coarse-surface Cp onto fine-surface face centers.

    Method:
        1. Use fine face centers as target points.
        2. Find the closest coarse triangle element to each fine face center.
        3. Project the fine face center onto that coarse triangle.
        4. Compute barycentric coordinates on the coarse triangle.
        5. Linearly interpolate coarse Cp from the triangle vertices.
        6. Compute Cp error on the fine surface.

    Parameters
    ----------
    fine_file : str
        Fine-resolution surface VTK file.

    coarse_file : str
        Coarse-resolution surface VTK file.

    output_file : str
        Output VTK file. Error fields are written on the fine mesh as cell data.

    field_name : str
        Name of the Cp field.

    distance_tol : float or None
        Maximum allowed projection distance. If None, it is estimated from the
        fine mesh area.

    normal_dot_tol : float
        Minimum allowed absolute dot product between fine and coarse normals.
        Use 0.7--0.9. Larger values are stricter.

    Returns
    -------
    metrics : dict
        Area-weighted error metrics.
    """
    fine_raw = pv.read(fine_file)
    coarse_raw = pv.read(coarse_file)

    fine = _as_tri_surface_with_cell_field(fine_raw, field_name)
    coarse = _as_tri_surface_with_point_field(coarse_raw, field_name)

    # Fine face centers and areas.
    fine_centers = fine.cell_centers().points

    fine_area_mesh = fine.compute_cell_sizes(length=False, area=True, volume=False)
    fine_area = fine_area_mesh.cell_data["Area"]

    fine_normals_mesh = fine.compute_normals(
        point_normals=False,
        cell_normals=True,
        consistent_normals=True,
        auto_orient_normals=False,
        inplace=False,
    )
    fine_normals = fine_normals_mesh.cell_data["Normals"]

    Cp_fine = np.asarray(fine.cell_data[field_name]).reshape(-1)

    # Coarse triangle connectivity.
    faces = coarse.faces.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise RuntimeError("Coarse mesh is not fully triangulated.")

    tri_conn = faces[:, 1:4]
    coarse_points = coarse.points
    Cp_coarse_point = np.asarray(coarse.point_data[field_name]).reshape(-1)

    # Find closest coarse triangle and closest point on that triangle.
    cell_ids, closest_points = coarse.find_closest_cell(
        fine_centers,
        return_closest_point=True,
    )

    cell_ids = np.asarray(cell_ids, dtype=np.int64)
    closest_points = np.asarray(closest_points)

    # Gather triangle vertices.
    tri = tri_conn[cell_ids]
    a = coarse_points[tri[:, 0]]
    b = coarse_points[tri[:, 1]]
    c = coarse_points[tri[:, 2]]

    # Barycentric coordinates of projected point.
    lambdas = _triangle_barycentric_batch(closest_points, a, b, c)

    # Numerical safety: closest_points should be on the triangle, but tiny
    # negative values may appear from floating point roundoff.
    lambdas_clipped = np.clip(lambdas, 0.0, 1.0)
    lambda_sum = lambdas_clipped.sum(axis=1, keepdims=True)
    lambdas_clipped = lambdas_clipped / np.where(lambda_sum == 0.0, np.nan, lambda_sum)

    # Linear interpolation of coarse Cp.
    Cp_tri = Cp_coarse_point[tri]  # (M, 3)
    Cp_coarse_projected = np.sum(lambdas_clipped * Cp_tri, axis=1)

    # Projection distance.
    projection_distance = np.linalg.norm(fine_centers - closest_points, axis=1)

    # Coarse triangle normals.
    coarse_normals = np.cross(b - a, c - a)
    coarse_norm = np.linalg.norm(coarse_normals, axis=1, keepdims=True)
    coarse_normals = coarse_normals / np.maximum(coarse_norm, 1.0e-30)

    normal_dot = np.einsum("ij,ij->i", fine_normals, coarse_normals)

    # Automatic distance tolerance based on fine face size.
    if distance_tol is None:
        h = np.sqrt(np.median(fine_area[fine_area > 0.0]))
        distance_tol = 3.0 * h

    valid = np.isfinite(Cp_fine)
    valid &= np.isfinite(Cp_coarse_projected)
    valid &= np.all(np.isfinite(lambdas_clipped), axis=1)
    valid &= projection_distance <= distance_tol
    valid &= np.abs(normal_dot) >= normal_dot_tol
    valid &= fine_area > 0.0

    Cp_error = Cp_fine - Cp_coarse_projected

    # Store fields on the fine mesh as cell data.
    fine.cell_data["Cp_fine"] = Cp_fine
    fine.cell_data["Cp_coarse_projected"] = Cp_coarse_projected
    fine.cell_data["Cp_error"] = Cp_error
    fine.cell_data["abs_Cp_error"] = np.abs(Cp_error)
    fine.cell_data["valid_error_mask"] = valid.astype(np.int8)
    fine.cell_data["projection_distance"] = projection_distance
    fine.cell_data["normal_dot"] = normal_dot
    fine.cell_data["bary_lambda_0"] = lambdas_clipped[:, 0]
    fine.cell_data["bary_lambda_1"] = lambdas_clipped[:, 1]
    fine.cell_data["bary_lambda_2"] = lambdas_clipped[:, 2]

    # Area-weighted errors.
    A = fine_area[valid]
    e = Cp_error[valid]
    ref = Cp_fine[valid]

    if len(e) == 0:
        raise RuntimeError(
            "No valid projected cells. Increase distance_tol, reduce normal_dot_tol, "
            "or compare patches separately."
        )

    L1_area = np.sum(A * np.abs(e)) / np.sum(A)
    L2_area = np.sqrt(np.sum(A * e**2) / np.sum(A))
    Linf = np.max(np.abs(e))
    rel_L2_area = np.sqrt(np.sum(A * e**2) / np.sum(A * ref**2))
    valid_area_fraction = np.sum(A) / np.sum(fine_area)

    fine.save(output_file)

    metrics = {
        "L1_area": float(L1_area),
        "L2_area": float(L2_area),
        "relative_L2_area": float(rel_L2_area),
        "Linf": float(Linf),
        "valid_cells": int(valid.sum()),
        "n_cells": int(len(valid)),
        "valid_area_fraction": float(valid_area_fraction),
        "distance_tol": float(distance_tol),
        "normal_dot_tol": float(normal_dot_tol),
    }

    print("Saved:", output_file)
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return metrics




metrics = project_coarse_cp_to_fine_faces(
    fine_file="patch_L.vtk",
    coarse_file="patch_M.vtk",
    output_file="p_error_allPatches_projected.vtk",
    field_name="p",
    distance_tol=None,
    normal_dot_tol=0.8,
)
