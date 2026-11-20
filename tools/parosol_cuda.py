#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Run the native CUDA solver on a linear ParOSol HDF5 file."""

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import h5py
import numpy as np


STRAIN_COMPONENTS = ("e_xx", "e_yy", "e_zz", "e_xy", "e_xz", "e_yz")
BLOCK_ROWS = 65536
INT32_MAX = np.iinfo(np.int32).max


def read_scalar(group, name):
    values = np.asarray(group[name])
    if values.shape not in ((), (1,)) or values.dtype.kind not in "fiu":
        raise ValueError(f"{group.name}/{name} must contain a single real number")
    return float(values.item())


def check_boundary_conditions(group, name, image_shape, *, required):
    coordinates_key = f"{name}_Coordinates"
    values_key = f"{name}_Values"
    if not required and coordinates_key not in group and values_key not in group:
        return
    coordinates = group[coordinates_key][...]
    values = group[values_key][...]
    if (coordinates.ndim != 2 or coordinates.shape[1] != 4
            or coordinates.dtype.kind not in "iu"
            or values.shape != (len(coordinates),) or values.dtype.kind not in "fiu"):
        raise ValueError(f"{name} must contain (z, y, x, component) rows and one value per row")
    if (np.any(coordinates < 0) or np.any(coordinates[:, :3] > image_shape)
            or np.any(coordinates[:, 3] > 2) or not np.isfinite(values).all()
            or np.any(np.abs(values.astype(np.float64)) > np.finfo(np.float32).max)):
        raise ValueError(f"{name} contains an out-of-range coordinate, component or value")


def prepare_native_input(public_file, native_file):
    with h5py.File(public_file, "r") as public:
        group = public["Image_Data"]
        poisson_ratio = read_scalar(group, "Poisons_ratio")
        if not 0 <= poisson_ratio < 0.5:
            raise ValueError(f"Poisons_ratio must be in [0, 0.5) (got {poisson_ratio:g})")
        voxel_size = read_scalar(group, "Voxelsize")
        if not math.isfinite(voxel_size) or voxel_size <= 0:
            raise ValueError("Voxelsize must be positive and finite")

        source_image = group["Image"]
        if (source_image.ndim != 3 or source_image.dtype.kind not in "fiu"
                or min(source_image.shape) == 0 or max(source_image.shape) > 65535):
            raise ValueError("Image must be a real (z, y, x) modulus volume with axes in 1..65535")
        image = np.asarray(source_image, dtype=np.float32)
        if (not np.isfinite(image).all() or np.any(image < 0)
                or not 0 < float(image.max()) <= INT32_MAX):
            raise ValueError("Image must contain finite nonnegative MPa moduli in the material ID range")
        check_boundary_conditions(group, "Fixed_Displacement", image.shape, required=True)
        check_boundary_conditions(group, "Loaded_Nodes", image.shape, required=False)

        image_gpa = image / np.float32(1000)
        if np.any((image > 0) & (image_gpa == 0)):
            raise ValueError("a positive Image modulus underflows in the CUDA input's GPa representation")

        # The native executable modifies its input file in place. Give it
        # only the linear input data and keep the source file in MPa.
        with h5py.File(native_file, "w") as native:
            target = native.create_group("Image_Data")
            target.create_dataset("Image", data=image_gpa)
            for key in ("Voxelsize", "Poisons_ratio",
                        "Fixed_Displacement_Coordinates", "Fixed_Displacement_Values",
                        "Loaded_Nodes_Coordinates", "Loaded_Nodes_Values"):
                if key in group:
                    target.create_dataset(key, data=group[key][...])

    return image, voxel_size, poisson_ratio


def row_blocks(row_count):
    for start in range(0, row_count, BLOCK_ROWS):
        yield slice(start, min(start + BLOCK_ROWS, row_count))


def export_float_field(native, source, target, name, shape):
    field = native[source]
    if field.shape != shape or field.dtype.kind != "f":
        raise ValueError(f"unexpected CUDA output shape or type for {field.name}")
    for rows in row_blocks(shape[0]):
        if not np.isfinite(field[rows]).all():
            raise ValueError(f"nonfinite CUDA output in {field.name}")
    native.copy(field, target, name=name)


def export_cpu_results(public_file, native_file, image, voxel_size, poisson_ratio):
    element_count = np.count_nonzero(image)
    with h5py.File(native_file, "r") as native, h5py.File(public_file, "r+") as public:
        mesh = public.create_group("Mesh")
        solution = public.create_group("Solution")

        nodes = native["Mesh/Coordinates"]
        node_count = nodes.shape[0]
        if not 0 < node_count <= INT32_MAX:
            raise ValueError("CUDA output node count is outside the CPU connectivity range")
        export_float_field(native, "Mesh/Coordinates", mesh, "Coordinates", (node_count, 3))
        xyz = nodes[...]
        raw_elements = native["Mesh/Elements"]
        if raw_elements.shape != (element_count, 8) or raw_elements.dtype.kind not in "iu":
            raise ValueError("CUDA connectivity does not match the input volume")
        raw_modulus = native["Mesh/Emoduli"]
        raw_strain = [native[f"Solution/{name}"] for name in STRAIN_COMPONENTS]
        for field in (raw_modulus, *raw_strain):
            if field.shape != (element_count, 1) or field.dtype.kind != "f":
                raise ValueError(f"unexpected CUDA output shape or type for {field.name}")

        elements = mesh.create_dataset("Elements", (element_count, 8), dtype=np.int32)
        material_ids = mesh.create_dataset("Material IDs", (element_count, 1), dtype=np.int32)
        stress = solution.create_dataset("Element stress", (element_count, 6), dtype=np.float64)
        strain = solution.create_dataset("Element strain", (element_count, 6), dtype=np.float64)
        for rows in row_blocks(element_count):
            corners = raw_elements[rows]
            if np.any(corners < 0) or np.any(corners >= node_count):
                raise ValueError("CUDA connectivity references a missing node")
            elements[rows] = corners + 1

            # Corner 0 is the originating voxel's node in both solvers.
            # Look up the original MPa value to reproduce the CPU writer's
            # integer truncation, including at exact integer moduli.
            zyx = xyz[corners[:, 0], ::-1] / voxel_size
            rounded = np.rint(zyx)
            if (np.any(rounded < 0) or np.any(rounded >= image.shape)
                    or not np.allclose(zyx, rounded, atol=1e-7, rtol=0)):
                raise ValueError("CUDA element origin is outside the input voxel grid")
            origins = rounded.astype(np.intp)
            mpa = image[tuple(origins.T)]
            native_mpa = raw_modulus[rows, 0]
            if np.any(mpa <= 0) or not np.allclose(native_mpa, mpa, rtol=2e-7, atol=0):
                raise ValueError("CUDA element moduli differ from the input MPa moduli")
            material_ids[rows, 0] = mpa.astype(np.int32)

            components = np.column_stack([field[rows, 0] for field in raw_strain])
            if not np.isfinite(components).all():
                raise ValueError("nonfinite CUDA element strain")
            strain[rows] = components
            # At the element center, use the same isotropic linear law
            # and engineering shear convention (xx, yy, zz, xy, xz, yz)
            # as the CPU and CUDA postprocessors.
            shear = native_mpa / (2 * (1 + poisson_ratio))
            lame = 2 * shear * poisson_ratio / (1 - 2 * poisson_ratio)
            element_stress = shear[:, None] * components
            element_stress[:, :3] *= 2
            element_stress[:, :3] += lame[:, None] * components[:, :3].sum(axis=1)[:, None]
            if not np.isfinite(element_stress).all():
                raise ValueError("nonfinite CUDA element stress")
            stress[rows] = element_stress

        export_float_field(native, "Solution/disp", solution, "Nodal displacements", (node_count, 3))
        export_float_field(native, "Solution/force", solution, "Nodal forces", (node_count, 3))
        for name in ("SED", "EFF", "VonMises"):
            export_float_field(native, f"Solution/{name}", solution, name, (element_count, 1))


def run_native(input_file, solver, options):
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    subprocess.run([solver, *options, str(input_file)], env=env, check=True)


def solve(input_file, output_file, solver, options):
    if os.path.lexists(output_file):
        raise FileExistsError(f"output already exists: {output_file}")

    with tempfile.TemporaryDirectory(prefix=".parosol-cuda-", dir=output_file.parent) as staging:
        staging = Path(staging)
        public_file = staging / "result.h5"
        native_file = staging / "native.h5"
        with h5py.File(input_file, "r") as source, h5py.File(public_file, "w") as result:
            for name, value in source.attrs.items():
                result.attrs.create(name, value, dtype=source.attrs.get_id(name).dtype)
            for name in source:
                if name not in ("Mesh", "Solution"):
                    source.copy(name, result, expand_external=True)
        image, voxel_size, poisson_ratio = prepare_native_input(public_file, native_file)
        run_native(native_file, solver, options)
        export_cpu_results(public_file, native_file, image, voxel_size, poisson_ratio)
        # Atomically publish a closed result and refuse concurrent writers.
        os.link(public_file, output_file)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="linear HDF5 input in the selected format")
    parser.add_argument("output", nargs="?", type=Path, help="new HDF5 result for CPU format")
    parser.add_argument("--format", choices=("cpu", "native"), default="cpu",
                        help="CPU MPa format (default) or native GPa format with in-place output")
    parser.add_argument("--solver", help="path to the native cudaparosol executable")
    parser.add_argument("--level", type=int)
    parser.add_argument("--tol", type=float)
    parser.add_argument("--eigen-max-iterations", type=int)
    args = parser.parse_args(argv)

    if args.format == "native" and args.output is not None:
        parser.error("--format native writes to the input file in place and takes one path")
    if args.format == "cpu" and args.output is None:
        parser.error("CPU format requires an output HDF5 path")
    if args.level is not None and args.level < 1:
        parser.error("--level must be at least 1")
    if args.tol is not None and (not math.isfinite(args.tol) or args.tol <= 0):
        parser.error("--tol must be positive and finite")
    if args.eigen_max_iterations is not None and not 2 <= args.eigen_max_iterations <= 1000:
        parser.error("--eigen-max-iterations must be between 2 and 1000")

    local_binary = Path(__file__).resolve().with_name("cudaparosol")
    solver = args.solver or (str(local_binary) if local_binary.is_file() else "cudaparosol")
    options = []
    for key in ("level", "tol", "eigen_max_iterations"):
        value = getattr(args, key)
        if value is not None:
            options.extend(["--" + key.replace("_", "-"), str(value)])

    try:
        if args.format == "native":
            run_native(args.input, solver, options)
        else:
            solve(args.input, args.output, solver, options)
    except (KeyError, OSError, ValueError) as exc:
        print(f"parosol-cuda: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"parosol-cuda: solver exited with status {exc.returncode}", file=sys.stderr)
        return exc.returncode if exc.returncode > 0 else 128 - exc.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
