#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Check native CPU and CUDA results on the same public-format linear input."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import h5py
import numpy as np


ADAPTER = Path(__file__).resolve().parents[1] / "tools" / "parosol_cuda.py"
TOLERANCE = 1e-8
FIELD_TOLERANCE = 1e-5
SOLVER_OPTIONS = ["--level", "4", "--tol", str(TOLERANCE), "--eigen-max-iterations", "100"]
NODE_FIELDS = ("Nodal displacements", "Nodal forces")
ELEMENT_FIELDS = ("SED", "EFF", "VonMises", "Element strain", "Element stress")
NATIVE_STRAIN = ("e_xx", "e_yy", "e_zz", "e_xy", "e_xz", "e_yz")


def create_input(path, heterogeneous, poisson_ratio, *, native=False, depth=8, x_offset=0):
    shape = (depth, 9, 10) if heterogeneous else (depth, 8, 8)
    if heterogeneous:
        z, y, x = np.indices(shape)
        moduli = np.array([350, 512.25, 2500, 16000.5], dtype=np.float32)
        image = moduli[(3 * z + 2 * y + x + z * x + y * x) % len(moduli)]
        image[2:4, 2:4, 4:6] = 0
    else:
        image = np.full(shape, 1500, dtype=np.float32)

    nz, ny, nx = shape
    boundary = []
    values = []
    for y in range(ny + 1):
        for x in range(nx + 1):
            boundary.extend([(0, y, x + x_offset, axis) for axis in range(3)])
            boundary.append((nz, y, x + x_offset, 2))
            values.extend((0.0, 0.0, 0.0, -0.001))
    if x_offset:
        image = np.pad(image, ((0, 0), (0, 0), (x_offset, 0)))
    with h5py.File(path, "w") as file:
        group = file.create_group("Image_Data")
        group["Image"] = image / np.float32(1000) if native else image
        group["Voxelsize"] = np.array([0.125], dtype=np.float64)
        group["Poisons_ratio"] = np.float64(poisson_ratio) if native else [poisson_ratio]
        group["Fixed_Displacement_Coordinates"] = np.array(boundary, dtype=np.uint16)
        group["Fixed_Displacement_Values"] = np.array(values, dtype=np.float32)
        if heterogeneous:
            group["Loaded_Nodes_Coordinates"] = np.array([[nz, 3, 1 + x_offset, 0]], dtype=np.uint16)
            group["Loaded_Nodes_Values"] = np.array([0.02], dtype=np.float32)


def run(command, log, env, expected=0):
    with log.open("w") as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode != expected:
        raise RuntimeError(f"{command[0]} returned {result.returncode}, expected {expected}; see {log}")
    return log.read_text()


def join_rows(reference, actual):
    index = {tuple(row): i for i, row in enumerate(actual)}
    if len(index) != len(reference) or len(actual) != len(reference):
        raise ValueError("CPU and CUDA meshes contain different or duplicate rows")
    return np.array([index[tuple(row)] for row in reference], dtype=np.intp)


def compare_fields(cpu_file, cuda_file, input_file):
    with h5py.File(cpu_file, "r") as cpu, h5py.File(cuda_file, "r") as cuda, h5py.File(input_file, "r") as original:
        for name in original["Image_Data"]:
            np.testing.assert_array_equal(original[f"Image_Data/{name}"], cuda[f"Image_Data/{name}"])
        for group in ("Mesh", "Solution"):
            if set(cpu[group]) != set(cuda[group]):
                raise ValueError(f"CPU and CUDA {group} dataset names differ")
            for name in cpu[group]:
                if cpu[group][name].shape != cuda[group][name].shape:
                    raise ValueError(f"CPU and CUDA {group}/{name} shapes differ")
                if cpu[group][name].dtype != cuda[group][name].dtype:
                    raise ValueError(f"CPU and CUDA {group}/{name} types differ")

        reference_nodes = cpu["Mesh/Coordinates"][...]
        cuda_nodes = cuda["Mesh/Coordinates"][...]
        node_order = join_rows(reference_nodes, cuda_nodes)
        cpu_elements = cpu["Mesh/Elements"][...]
        cuda_elements = cuda["Mesh/Elements"][...]
        np.testing.assert_array_equal(
            [(elements.min(), elements.max()) for elements in (cpu_elements, cuda_elements)],
            [(1, len(reference_nodes))] * 2,
        )
        element_order = join_rows(reference_nodes[cpu_elements[:, 0] - 1],
                                  cuda_nodes[cuda_elements[:, 0] - 1])
        np.testing.assert_array_equal(reference_nodes[cpu_elements - 1],
                                      cuda_nodes[cuda_elements[element_order] - 1])
        np.testing.assert_array_equal(cpu["Mesh/Material IDs"][...],
                                      cuda["Mesh/Material IDs"][...][element_order])

        errors = {}
        for name in (*NODE_FIELDS, *ELEMENT_FIELDS):
            reference = cpu[f"Solution/{name}"][...]
            order = node_order if name in NODE_FIELDS else element_order
            actual = cuda[f"Solution/{name}"][...][order]
            scale = np.max(np.abs(reference))
            maximum = np.max(np.abs(actual - reference))
            if not np.isfinite(maximum) or maximum > FIELD_TOLERANCE * scale:
                raise ValueError(f"{name} maximum error {maximum:g} exceeds {FIELD_TOLERANCE:g} × {scale:g}")
            errors[name] = float(maximum / scale) if scale else 0.0

        reactions = {}
        at_bottom = reference_nodes[:, 2] == 0
        at_top = reference_nodes[:, 2] == reference_nodes[:, 2].max()
        for label, file, order in (("cpu", cpu, slice(None)), ("cuda", cuda, node_order)):
            forces = file["Solution/Nodal forces"][...][order]
            net_force = float(np.linalg.norm(forces.sum(axis=0)))
            if net_force > 1e-10 * np.abs(forces).sum():
                raise ValueError(f"{label} net force is {net_force:g} N")
            if "Loaded_Nodes_Values" in original["Image_Data"]:
                loaded_node = (reference_nodes == (0.125, 0.375, 1.0)).all(axis=1)
                np.testing.assert_allclose(forces[loaded_node, 0], [0.02], atol=2e-8, rtol=0)
            reactions[label] = {
                "bottom_z_N": float(forces[at_bottom, 2].sum()),
                "top_z_N": float(forces[at_top, 2].sum()),
                "net_norm_N": net_force,
            }
        return {"nodes": len(reference_nodes), "elements": len(cpu_elements),
                "max_error_over_peak_cpu": errors, "reactions": reactions}


def check_native_format(base, adapted, native, input_image, poisson_ratio, x_offset=0):
    with h5py.File(base, "r") as public_input, h5py.File(adapted, "r") as public, h5py.File(native, "r") as raw:
        np.testing.assert_array_equal(raw["Image_Data/Image"], input_image)
        np.testing.assert_equal(raw["Image_Data/Poisons_ratio"][...], np.float64(poisson_ratio))
        for name in public_input["Image_Data"]:
            if name not in ("Image", "Poisons_ratio"):
                expected = public_input[f"Image_Data/{name}"][...]
                if name.endswith("_Coordinates"):
                    expected[:, 2] += x_offset
                np.testing.assert_array_equal(raw[f"Image_Data/{name}"], expected)
        raw_points = raw["Mesh/Coordinates"][...]
        raw_points[:, 0] -= x_offset * public_input["Image_Data/Voxelsize"][0]
        public_points = public["Mesh/Coordinates"][...]
        node_order = join_rows(public_points, raw_points)
        raw_cells = raw["Mesh/Elements"][...]
        public_cells = public["Mesh/Elements"][...] - 1
        np.testing.assert_equal((raw_cells.min(), raw_cells.max()), (0, len(raw_points) - 1))
        cell_order = join_rows(public_points[public_cells[:, 0]], raw_points[raw_cells[:, 0]])
        np.testing.assert_array_equal(raw_points[raw_cells[cell_order]], public_points[public_cells])

        for raw_name, public_name in (("disp", "Nodal displacements"), ("force", "Nodal forces")):
            np.testing.assert_allclose(raw[f"Solution/{raw_name}"][...][node_order],
                                       public[f"Solution/{public_name}"], atol=1e-12, rtol=1e-9)
        for name in ("SED", "EFF", "VonMises"):
            np.testing.assert_allclose(raw[f"Solution/{name}"][...][cell_order],
                                       public[f"Solution/{name}"], atol=1e-12, rtol=1e-9)
        strain = np.column_stack([raw[f"Solution/{name}"][:, 0] for name in NATIVE_STRAIN])
        np.testing.assert_allclose(strain[cell_order], public["Solution/Element strain"],
                                   atol=1e-12, rtol=1e-9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", required=True, type=Path)
    parser.add_argument("--cuda", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--case", choices=("homogeneous", "heterogeneous", "five-slice"))
    parser.add_argument("--poisson-ratio", type=float, default=0.3)
    args = parser.parse_args()
    if not 0 <= args.poisson_ratio < 0.5:
        parser.error("--poisson-ratio must be in [0, 0.5)")
    work = args.work_dir.resolve()
    work.mkdir()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="4", OMP_THREAD_LIMIT="4")
    results = {}
    for name in ("homogeneous", "heterogeneous", "five-slice"):
        if args.case and name != args.case:
            continue
        heterogeneous = name == "heterogeneous"
        depth = 5 if name == "five-slice" else 8
        # Translate the native copy to exercise unsigned 16-bit boundary coordinates.
        x_offset = 32768 if name == "five-slice" else 0
        base = work / f"{name}-input.h5"
        reference = work / f"{name}-cpu.h5"
        output = work / f"{name}-cuda.h5"
        native = work / f"{name}-native.h5"
        create_input(base, heterogeneous, args.poisson_ratio, depth=depth)
        create_input(native, heterogeneous, args.poisson_ratio, depth=depth, native=True, x_offset=x_offset)
        digest = hashlib.sha256(base.read_bytes()).hexdigest()
        shutil.copyfile(base, reference)
        log = run([str(args.cpu.resolve()), *SOLVER_OPTIONS, str(reference)], work / f"{name}-cpu.log", env)
        residuals = [float(value) for value in re.findall(r"(?m)^Residuum (?:before|after) solving:\s*(\S+)", log)]
        if len(residuals) != 2 or not 0 <= residuals[1] <= TOLERANCE * residuals[0]:
            raise ValueError(f"CPU did not reach the requested tolerance for {name}")

        command = [sys.executable, str(ADAPTER), "--solver", str(args.cuda.resolve()), *SOLVER_OPTIONS]
        log = run([*command, str(base), str(output)], work / f"{name}-cuda.log", env)
        status = json.loads(re.search(r"(?m)^PAROSOL_RESULT (.+)$", log)[1])
        if not status["converged"] or status["relative_residual"] > TOLERANCE:
            raise ValueError(f"CUDA did not reach the requested tolerance for {name}")
        if hashlib.sha256(base.read_bytes()).hexdigest() != digest:
            raise ValueError("CUDA adapter modified the source HDF5")
        results[name] = compare_fields(reference, output, base)

        with h5py.File(native, "r+") as model:
            input_image = model["Image_Data/Image"][...]
            if name == "five-slice":
                # Existing output can belong to a previous mesh with a different number of elements.
                model["Mesh/Elements"] = np.zeros((np.count_nonzero(input_image) + 1, 8), dtype=np.int64)
                model["Solution/SED"] = np.zeros((1, 1))
        inode = native.stat().st_ino
        run([*command, "--format", "native", str(native)], work / f"{name}-native.log", env)
        if native.stat().st_ino != inode:
            raise ValueError("native HDF5 file was replaced")
        check_native_format(base, output, native, input_image, args.poisson_ratio, x_offset)

    for label, value in (("out-of-range", 0.5), ("nonscalar", [0.25, 0.3])):
        invalid = work / f"invalid-poisson-{label}.h5"
        create_input(invalid, False, args.poisson_ratio, native=True)
        with h5py.File(invalid, "r+") as file:
            del file["Image_Data/Poisons_ratio"]
            file["Image_Data/Poisons_ratio"] = value
        original = hashlib.sha256(invalid.read_bytes()).hexdigest()
        log = run([str(args.cuda.resolve()), str(invalid)], work / f"invalid-poisson-{label}.log",
                  env, expected=2)
        if "/Image_Data/Poisons_ratio must be a single number in [0, 0.5)" not in log:
            raise ValueError("native solver did not report the invalid Poisson ratio")
        if hashlib.sha256(invalid.read_bytes()).hexdigest() != original:
            raise ValueError("native solver modified a file with invalid Poisson ratio")
    (work / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
