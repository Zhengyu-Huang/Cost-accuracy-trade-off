#!/usr/bin/env python3
"""Prepare a combined DrivAerNet++ STL for OpenFOAM 12.

Processing order:

1. read the binary or ASCII STL and merge coincident STL vertices;
2. split the surface into one body and four wheel components;
3. make adjacent triangle winding consistent within each component;
4. orient every closed component outward using its signed volume;
5. group the two front and two rear wheels using an x-coordinate threshold;
6. write exactly ``body.obj.gz``, ``frontWheels.obj.gz``, and
   ``rearWheels.obj.gz``.

The script also reports front/rear axle origins, wheel radii, ground
clearances, and wheelbase for updating OpenFOAM's ``0/U`` file.  Coordinates
are never scaled or translated.

Run with ParaView's Python runtime, for example::

    /Applications/ParaView-5.11.2.app/Contents/bin/pvpython \
        scripts/split_drivaer_stl.py E_S_WWC_WM_005.stl \
        -o output/constant/geometry
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

try:
    import numpy as np
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except ImportError as exc:  # pragma: no cover - depends on local runtime
    raise SystemExit(
        "NumPy and VTK are required. Run this script with ParaView's "
        "pvpython or install the Python 'numpy' and 'vtk' packages."
    ) from exc


OUTPUT_PARTS = ("body", "frontWheels", "rearWheels")


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_stl(path: Path) -> Any:
    """Read an STL, merge shared vertices, and require triangle cells."""

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(path))
    reader.MergingOn()
    reader.Update()
    mesh = reader.GetOutput()

    if mesh is None or mesh.GetNumberOfPoints() == 0:
        raise RuntimeError(f"STL reader returned no points for {path}")
    if mesh.GetNumberOfPolys() == 0:
        raise RuntimeError(f"STL reader returned no faces for {path}")
    if mesh.GetNumberOfCells() != mesh.GetNumberOfPolys():
        raise RuntimeError("The STL contains non-polygonal cells")

    for cell_id in range(mesh.GetNumberOfCells()):
        if mesh.GetCellType(cell_id) != vtk.VTK_TRIANGLE:
            raise RuntimeError(f"STL cell {cell_id} is not a triangle")

    result = vtk.vtkPolyData()
    result.DeepCopy(mesh)
    return result


def compact_points(polydata: Any) -> Any:
    """Remove unused points without merging distinct surface vertices."""

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(polydata)
    cleaner.PointMergingOff()
    cleaner.ConvertLinesToPointsOff()
    cleaner.ConvertPolysToLinesOff()
    cleaner.ConvertStripsToPolysOff()
    cleaner.Update()

    result = vtk.vtkPolyData()
    result.DeepCopy(cleaner.GetOutput())
    return result


def reverse_triangle_winding(polydata: Any) -> Any:
    """Reverse every triangle from (i,j,k) to (i,k,j)."""

    triangles = vtk.vtkCellArray()
    point_ids = vtk.vtkIdList()
    polygons = polydata.GetPolys()
    polygons.InitTraversal()
    while polygons.GetNextCell(point_ids):
        if point_ids.GetNumberOfIds() != 3:
            raise RuntimeError("Normal reversal requires a triangle-only surface")
        triangles.InsertNextCell(3)
        triangles.InsertCellPoint(point_ids.GetId(0))
        triangles.InsertCellPoint(point_ids.GetId(2))
        triangles.InsertCellPoint(point_ids.GetId(1))

    result = vtk.vtkPolyData()
    result.DeepCopy(polydata)
    result.SetPolys(triangles)
    result.Modified()
    return result


def signed_volume(polydata: Any) -> float:
    """Compute oriented volume of a closed triangle surface."""

    points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(
        np.float64, copy=False
    )
    raw_faces = vtk_to_numpy(polydata.GetPolys().GetData()).astype(
        np.int64, copy=False
    )
    if raw_faces.size != 4 * polydata.GetNumberOfPolys():
        raise RuntimeError("Expected legacy triangle connectivity [3,i,j,k]")
    packed = raw_faces.reshape((-1, 4))
    if not np.all(packed[:, 0] == 3):
        raise RuntimeError("Signed volume requires a triangle-only surface")
    faces = packed[:, 1:]
    p0 = points[faces[:, 0]]
    p1 = points[faces[:, 1]]
    p2 = points[faces[:, 2]]
    return float(np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() / 6.0)


def count_feature_edges(polydata: Any, *, boundary: bool, non_manifold: bool) -> int:
    """Count selected feature-edge types."""

    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputData(polydata)
    feature_edges.FeatureEdgesOff()
    feature_edges.ManifoldEdgesOff()
    if boundary:
        feature_edges.BoundaryEdgesOn()
    else:
        feature_edges.BoundaryEdgesOff()
    if non_manifold:
        feature_edges.NonManifoldEdgesOn()
    else:
        feature_edges.NonManifoldEdgesOff()
    feature_edges.ColoringOff()
    feature_edges.Update()
    return int(feature_edges.GetOutput().GetNumberOfCells())


def component_stats(region_id: int, polydata: Any) -> dict[str, Any]:
    """Return geometry and topology statistics for one connected component."""

    bounds = [float(value) for value in polydata.GetBounds()]
    return {
        "region_id": region_id,
        "n_points": int(polydata.GetNumberOfPoints()),
        "n_triangles": int(polydata.GetNumberOfPolys()),
        "bounds": bounds,
        "bounds_center": [
            0.5 * (bounds[0] + bounds[1]),
            0.5 * (bounds[2] + bounds[3]),
            0.5 * (bounds[4] + bounds[5]),
        ],
        "boundary_edges": count_feature_edges(
            polydata, boundary=True, non_manifold=False
        ),
        "non_manifold_edges": count_feature_edges(
            polydata, boundary=False, non_manifold=True
        ),
        "signed_volume": signed_volume(polydata),
    }


def split_connected_regions(polydata: Any) -> list[dict[str, Any]]:
    """Extract and compact every connected triangle component."""

    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(polydata)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.ColorRegionsOff()
    connectivity.Update()
    count = int(connectivity.GetNumberOfExtractedRegions())

    components: list[dict[str, Any]] = []
    for region_id in range(count):
        connectivity.SetExtractionModeToSpecifiedRegions()
        connectivity.InitializeSpecifiedRegionList()
        connectivity.AddSpecifiedRegion(region_id)
        connectivity.Modified()
        connectivity.Update()

        region = compact_points(connectivity.GetOutput())
        stats = component_stats(region_id, region)
        components.append({"mesh": region, "stats": stats})
    return components


def make_winding_consistent(polydata: Any) -> Any:
    """Propagate a consistent winding through one connected component."""

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.ComputePointNormalsOff()
    normals.ComputeCellNormalsOn()
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOff()
    normals.NonManifoldTraversalOn()
    normals.Update()

    result = vtk.vtkPolyData()
    result.DeepCopy(normals.GetOutput())
    # OBJ/OpenFOAM orientation is defined by face winding, not stored normals.
    result.GetPointData().Initialize()
    result.GetCellData().Initialize()
    result.GetFieldData().Initialize()
    return result


def orient_components_outward(
    components: list[dict[str, Any]], volume_tolerance: float
) -> list[dict[str, Any]]:
    """Make each closed component consistently outward-oriented."""

    for item in components:
        stats = item["stats"]
        if stats["boundary_edges"] != 0 or stats["non_manifold_edges"] != 0:
            raise RuntimeError(
                f"Region {stats['region_id']} is not a closed manifold: "
                f"boundary={stats['boundary_edges']}, "
                f"non-manifold={stats['non_manifold_edges']}"
            )

        volume_before_consistency = float(stats["signed_volume"])
        item["mesh"] = make_winding_consistent(item["mesh"])
        before_global_orientation = signed_volume(item["mesh"])
        if abs(before_global_orientation) <= volume_tolerance:
            raise RuntimeError(
                f"Region {stats['region_id']} has near-zero signed volume "
                f"({before_global_orientation:.9g}); outward orientation is "
                "ambiguous"
            )

        flipped = before_global_orientation < 0.0
        if flipped:
            item["mesh"] = reverse_triangle_winding(item["mesh"])
        after = signed_volume(item["mesh"])
        if after <= volume_tolerance:
            raise RuntimeError(
                f"Failed to orient region {stats['region_id']} outward; "
                f"signed volume is {after:.9g}"
            )
        stats["signed_volume_before_consistency"] = volume_before_consistency
        stats["signed_volume_after_consistency"] = before_global_orientation
        stats["outward_correction_applied"] = flipped
        stats["signed_volume"] = after
    return components


def classify_components(
    components: list[dict[str, Any]],
    x_split: float,
    ground_z: float,
    ground_tolerance: float,
) -> dict[str, list[dict[str, Any]]]:
    """Use z=ground for wheels, then split front/rear using x_split."""

    if len(components) != 5:
        raise RuntimeError(
            f"Expected five components (body plus four wheels), found "
            f"{len(components)}"
        )

    wheels = [
        item
        for item in components
        if abs(float(item["stats"]["bounds"][4]) - ground_z)
        <= ground_tolerance
    ]
    wheel_object_ids = {id(item) for item in wheels}
    body_candidates = [
        item for item in components if id(item) not in wheel_object_ids
    ]
    if len(wheels) != 4 or len(body_candidates) != 1:
        wheel_ids = [item["stats"]["region_id"] for item in wheels]
        raise RuntimeError(
            "Expected four wheel components at the ground and one remaining "
            f"body; found wheel regions {wheel_ids} and "
            f"{len(body_candidates)} body candidates"
        )
    body = body_candidates[0]

    front = [
        item
        for item in wheels
        if float(item["stats"]["bounds_center"][0]) < x_split
    ]
    rear = [
        item
        for item in wheels
        if float(item["stats"]["bounds_center"][0]) > x_split
    ]
    if len(front) != 2 or len(rear) != 2:
        raise RuntimeError(
            f"Expected two front and two rear wheels using x_split={x_split:g}; "
            f"found {len(front)} front and {len(rear)} rear"
        )
    return {"body": [body], "frontWheels": front, "rearWheels": rear}


def append_parts(parts: list[dict[str, Any]]) -> Any:
    """Append confirmed components without changing coordinates or winding."""

    appender = vtk.vtkAppendPolyData()
    for part in parts:
        appender.AddInputData(part["mesh"])
    appender.Update()
    result = compact_points(appender.GetOutput())
    result.GetPointData().Initialize()
    result.GetCellData().Initialize()
    result.GetFieldData().Initialize()
    return result


def wheel_group_geometry(
    wheels: list[dict[str, Any]], ground_z: float
) -> dict[str, Any]:
    """Estimate an axle origin and tire radius from a left/right wheel pair."""

    if len(wheels) != 2:
        raise RuntimeError(f"Expected a wheel pair, received {len(wheels)} regions")
    individual: list[dict[str, Any]] = []
    for wheel in sorted(
        wheels, key=lambda item: float(item["stats"]["bounds_center"][1])
    ):
        stats = wheel["stats"]
        bounds = [float(value) for value in stats["bounds"]]
        center = [float(value) for value in stats["bounds_center"]]
        individual.append(
            {
                "region_id": int(stats["region_id"]),
                "side": "negative-y" if center[1] < 0.0 else "positive-y",
                "bounds_center": center,
                "vertical_radius": 0.5 * (bounds[5] - bounds[4]),
                "streamwise_half_extent": 0.5 * (bounds[1] - bounds[0]),
                "ground_clearance": bounds[4] - ground_z,
            }
        )

    origin = [
        sum(item["bounds_center"][0] for item in individual) / 2.0,
        0.0,
        sum(item["bounds_center"][2] for item in individual) / 2.0,
    ]
    geometric_radius = sum(
        item["vertical_radius"] for item in individual
    ) / 2.0
    return {
        "origin": origin,
        "geometric_radius": geometric_radius,
        "effective_rolling_radius": origin[2] - ground_z,
        "ground_clearance": sum(
            item["ground_clearance"] for item in individual
        )
        / 2.0,
        "individual_wheels": individual,
    }


def estimate_wheel_geometry(
    assignments: dict[str, list[dict[str, Any]]], ground_z: float
) -> dict[str, Any]:
    """Estimate front/rear axle origins, radii, clearances, and wheelbase."""

    front = wheel_group_geometry(assignments["frontWheels"], ground_z)
    rear = wheel_group_geometry(assignments["rearWheels"], ground_z)
    return {
        "front": front,
        "rear": rear,
        "wheelbase": rear["origin"][0] - front["origin"][0],
        "average_geometric_radius": 0.5
        * (front["geometric_radius"] + rear["geometric_radius"]),
        "average_effective_rolling_radius": 0.5
        * (
            front["effective_rolling_radius"]
            + rear["effective_rolling_radius"]
        ),
    }


def print_wheel_geometry(geometry: dict[str, Any]) -> None:
    """Print wheel parameters in an OpenFOAM-friendly form."""

    front = geometry["front"]
    rear = geometry["rear"]
    print("Estimated wheel parameters from STL bounds (metres):")
    print(
        "  front axle origin  "
        f"({front['origin'][0]:.9g} 0 {front['origin'][2]:.9g})"
    )
    print(
        "  front rolling radius (0/U) "
        f"{front['effective_rolling_radius']:.9g}"
    )
    print(f"  front geometric radius     {front['geometric_radius']:.9g}")
    print(f"  front clearance    {front['ground_clearance']:.9g}")
    print(
        "  rear axle origin   "
        f"({rear['origin'][0]:.9g} 0 {rear['origin'][2]:.9g})"
    )
    print(
        "  rear rolling radius  (0/U) "
        f"{rear['effective_rolling_radius']:.9g}"
    )
    print(f"  rear geometric radius      {rear['geometric_radius']:.9g}")
    print(f"  rear clearance     {rear['ground_clearance']:.9g}")
    print(f"  wheelbase          {geometry['wheelbase']:.9g}")
    print(
        "  average geometric radius    "
        f"{geometry['average_geometric_radius']:.9g}"
    )
    print(
        "  average rolling radius      "
        f"{geometry['average_effective_rolling_radius']:.9g}"
    )


def topology_stats(polydata: Any, area_tolerance: float) -> dict[str, Any]:
    """Check triangle areas and surface topology."""

    points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(
        np.float64, copy=False
    )
    raw_faces = vtk_to_numpy(polydata.GetPolys().GetData()).astype(
        np.int64, copy=False
    )
    packed = raw_faces.reshape((-1, 4))
    if not np.all(packed[:, 0] == 3):
        raise RuntimeError("Output contains a non-triangle face")
    faces = packed[:, 1:]
    p0 = points[faces[:, 0]]
    p1 = points[faces[:, 1]]
    p2 = points[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)

    return {
        "n_points": int(polydata.GetNumberOfPoints()),
        "n_triangles": int(polydata.GetNumberOfPolys()),
        "triangles_at_or_below_area_tolerance": int(
            np.count_nonzero(areas <= area_tolerance)
        ),
        "area_tolerance": area_tolerance,
        "minimum_triangle_area": float(areas.min()) if areas.size else 0.0,
        "boundary_edges": count_feature_edges(
            polydata, boundary=True, non_manifold=False
        ),
        "non_manifold_edges": count_feature_edges(
            polydata, boundary=False, non_manifold=True
        ),
        "signed_volume": signed_volume(polydata),
        "bounds": [float(value) for value in polydata.GetBounds()],
    }


def write_obj_gz(polydata: Any, target: Path) -> None:
    """Write an ASCII triangle OBJ and gzip it deterministically."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".obj", dir=str(target.parent)
    )
    os.close(descriptor)
    temporary_obj = Path(temporary_name)
    temporary_gz = temporary_obj.with_suffix(".obj.gz.tmp")
    try:
        writer = vtk.vtkOBJWriter()
        writer.SetFileName(str(temporary_obj))
        writer.SetInputData(polydata)
        if writer.Write() != 1:
            raise RuntimeError(f"VTK failed to write {temporary_obj}")

        with temporary_obj.open("rb") as source, temporary_gz.open("wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
                shutil.copyfileobj(source, gz, length=1024 * 1024)
        os.replace(temporary_gz, target)
    finally:
        temporary_obj.unlink(missing_ok=True)
        temporary_gz.unlink(missing_ok=True)


def validate_obj_gz(path: Path, expected_vertices: int, expected_faces: int) -> None:
    """Verify gzip integrity and require triangle-only ASCII OBJ records."""

    vertices = 0
    faces = 0
    with gzip.open(path, "rt", encoding="ascii") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.startswith("v "):
                vertices += 1
            elif line.startswith("f "):
                if len(line.split()) != 4:
                    raise RuntimeError(
                        f"{path.name}:{line_number} is not a triangular OBJ face"
                    )
                faces += 1
    if vertices != expected_vertices or faces != expected_faces:
        raise RuntimeError(
            f"Validation failed for {path.name}: expected {expected_vertices}/"
            f"{expected_faces} vertices/faces, read {vertices}/{faces}"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Flip, split, outward-orient, and export a five-component "
            "DrivAerNet++ STL as the three OBJ.GZ files required by OpenFOAM."
        )
    )
    parser.add_argument("input", type=Path, help="Input binary or ASCII STL")
    parser.add_argument(
        "-o", "--output-dir", type=Path,
        help="Output directory (default: ./<input-stem>_openfoam_geometry)",
    )
    parser.add_argument(
        "--x-split", type=float, default=1.0,
        help="Wheel-centre x threshold separating front/rear (default: 1.0)",
    )
    parser.add_argument(
        "--ground-z", type=float, default=0.0,
        help="Ground-plane z coordinate (default: 0.0)",
    )
    parser.add_argument(
        "--ground-tolerance", type=float, default=1.0e-3,
        help="Maximum wheel z_min distance from ground (default: 1e-3 m)",
    )
    parser.add_argument(
        "--volume-tolerance", type=float, default=1.0e-12,
        help="Near-zero signed-volume threshold (default: 1e-12 m^3)",
    )
    parser.add_argument(
        "--area-tolerance", type=float, default=1.0e-14,
        help="Small-triangle area threshold (default: 1e-14 m^2)",
    )
    parser.add_argument(
        "--initial-global-flip", action="store_true",
        help=(
            "Reverse the whole STL before automatic per-component orientation; "
            "intended only for diagnostics because auto mode does not need it"
        ),
    )
    parser.add_argument(
        "--manifest", type=Path,
        help="Optional JSON diagnostics path; geometry output remains three files",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Replace existing generated outputs",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Inspect and classify without writing output files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path.suffix.lower() != ".stl":
        raise ValueError(f"Expected an .stl input, received {input_path.name}")
    if (
        args.ground_tolerance < 0
        or args.volume_tolerance < 0
        or args.area_tolerance < 0
    ):
        raise ValueError("Tolerances must be non-negative")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (Path.cwd() / f"{input_path.stem}_openfoam_geometry").resolve()
    )

    source = read_stl(input_path)
    input_summary = {
        "n_points": int(source.GetNumberOfPoints()),
        "n_triangles": int(source.GetNumberOfPolys()),
        "bounds": [float(value) for value in source.GetBounds()],
    }
    initial_flip_applied = args.initial_global_flip
    if initial_flip_applied:
        source = reverse_triangle_winding(source)
        print(
            f"Optional initial global winding flip: "
            f"{source.GetNumberOfPolys()} triangles"
        )

    components = split_connected_regions(source)
    components = orient_components_outward(components, args.volume_tolerance)
    assignments = classify_components(
        components,
        x_split=args.x_split,
        ground_z=args.ground_z,
        ground_tolerance=args.ground_tolerance,
    )
    wheel_geometry = estimate_wheel_geometry(assignments, args.ground_z)

    labels_by_region: dict[int, str] = {}
    for label, items in assignments.items():
        for item in items:
            labels_by_region[int(item["stats"]["region_id"])] = label

    print(f"Input: {input_path}")
    print(
        f"Surface: {source.GetNumberOfPoints()} points, "
        f"{source.GetNumberOfPolys()} triangles, {len(components)} regions"
    )
    for item in sorted(components, key=lambda entry: entry["stats"]["region_id"]):
        stats = item["stats"]
        center = stats["bounds_center"]
        print(
            f"region={stats['region_id']:>2}  "
            f"part={labels_by_region[stats['region_id']]:<11}  "
            f"triangles={stats['n_triangles']:>8}  "
            f"center=({center[0]: .6f}, {center[1]: .6f}, {center[2]: .6f})  "
            f"outward_fix={stats['outward_correction_applied']}"
        )
    print_wheel_geometry(wheel_geometry)

    prepared_meshes: dict[str, Any] = {}
    outputs: dict[str, Any] = {}
    warnings: list[str] = []
    for label in OUTPUT_PARTS:
        prepared = append_parts(assignments[label])
        quality = topology_stats(prepared, args.area_tolerance)
        prepared_meshes[label] = prepared
        outputs[label] = quality
        if quality["triangles_at_or_below_area_tolerance"]:
            warnings.append(
                f"{label}: {quality['triangles_at_or_below_area_tolerance']} "
                "degenerate/tiny triangles"
            )
        if quality["boundary_edges"]:
            warnings.append(f"{label}: {quality['boundary_edges']} boundary edges")
        if quality["non_manifold_edges"]:
            warnings.append(
                f"{label}: {quality['non_manifold_edges']} non-manifold edges"
            )
        if quality["signed_volume"] <= args.volume_tolerance:
            warnings.append(
                f"{label}: non-positive/near-zero signed volume "
                f"{quality['signed_volume']:.9g}"
            )

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if args.dry_run:
        print("Dry run complete; no files were written.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        label: output_dir / f"{label}.obj.gz" for label in OUTPUT_PARTS
    }
    manifest_path = (
        args.manifest.expanduser().resolve() if args.manifest is not None else None
    )
    protected = [*targets.values()]
    if manifest_path is not None:
        protected.append(manifest_path)
    existing = [path for path in protected if path.exists()]
    if existing and not args.force:
        formatted = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to replace existing output. Use --force if intended:\n  "
            + formatted
        )

    for label, target in targets.items():
        write_obj_gz(prepared_meshes[label], target)
        validate_obj_gz(
            target,
            expected_vertices=outputs[label]["n_points"],
            expected_faces=outputs[label]["n_triangles"],
        )
        outputs[label]["path"] = str(target)
        outputs[label]["bytes"] = target.stat().st_size
        outputs[label]["sha256"] = sha256(target)
        print(
            f"Wrote {target.name}: {outputs[label]['n_triangles']} triangles, "
            f"{target.stat().st_size / (1024 * 1024):.2f} MiB"
        )

    if manifest_path is not None:
        manifest = {
            "source": {
                "path": str(input_path),
                "bytes": input_path.stat().st_size,
                "sha256": sha256(input_path),
                **input_summary,
            },
            "vtk_version": vtk.vtkVersion.GetVTKVersion(),
            "processing": {
                "normal_orientation_mode": "automatic per closed component",
                "initial_global_flip_applied": initial_flip_applied,
                "x_split": args.x_split,
                "ground_z": args.ground_z,
                "ground_tolerance": args.ground_tolerance,
                "components": [item["stats"] for item in components],
                "assignments": {
                    label: [
                        item["stats"]["region_id"] for item in assignments[label]
                    ]
                    for label in OUTPUT_PARTS
                },
            },
            "wheel_geometry": wheel_geometry,
            "outputs": outputs,
            "warnings": warnings,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote diagnostics manifest: {manifest_path}")

    print(
        "Next: run OpenFOAM surfaceCheck on all three surfaces and update "
        "the front/rear wheel origins and radii in 0/U."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
