# CUDA ParOSol

The CUDA build adds a standalone solver and an adapter for the [linear ParOSol HDF5 format](file_format.md). The existing CPU executable and build are the defaults.

## Build on Linux

Use the CPU build dependencies, a CUDA toolkit and driver suitable for your GPU, and a host compiler with OpenMP support. The following example builds for compute capability 8.9. Set the architecture list to match the GPUs on which you will run.

~~~sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DPAROSOL_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda --parallel
ctest --test-dir build-cuda --output-on-failure
cmake --install build-cuda --prefix "$PWD/local"
~~~

The adapter uses Python 3, NumPy and h5py. CMake adds its tests when the selected Python interpreter can import both packages.

## Run an existing CPU input file on a GPU

Make one GPU visible to the process. Use a single process and set the number of OpenMP threads to 1, 2, 4, 8 or 16. The adapter sets one thread when `OMP_NUM_THREADS` is unset. The native executable checks both the requested thread count and the actual team before CUDA initialization.

~~~sh
OMP_NUM_THREADS=4 local/bin/parosol-cuda existing-cpu-input.h5 cuda-result.h5
~~~

The output path must be new. The adapter writes a new result after the solve succeeds. It copies the input datasets, including externally linked HDF5 datasets, into that result. It accepts `--level`, `--tol`, and `--eigen-max-iterations`. To run it without installation, use `python3 tools/parosol_cuda.py` and pass `--solver build-cuda/src/parOsol_cuda/src/cudaparosol`.

Both input modes support linear isotropic elasticity with cubic voxels. Set a single Poisson's ratio in `/Image_Data/Poisons_ratio`, in the range `0 <= nu < 0.5`. CUDA uses this value for fine and coarse stiffness, element stresses, and strain energy density. Step-based ParOSol-NL inputs use a separate format.

| Data | CPU-facing file | Native CUDA job copy |
| --- | --- | --- |
| Input `/Image_Data/Image` | Young’s modulus in MPa, in `(z, y, x)` order | Modulus divided by 1,000 to GPa |
| Voxel size and boundary conditions | mm and N, using `(z, y, x, component)` node indices | Same values and indices |
| `/Mesh/Elements` | One-based node IDs | Zero-based node IDs |
| `/Mesh/Material IDs` | Integer truncation of the original MPa value, as in the CPU writer | `/Mesh/Emoduli` exports floating-point MPa values |
| Nodal results | `Nodal displacements` in mm and `Nodal forces` in N | `disp` and `force` in the same units |
| Element results | `SED`, `EFF`, `VonMises`, `Element strain` and `Element stress` | Same scalar fields and six separate strain arrays |

The adapter preserves the original modulus volume in the result. `SED`, `VonMises` and the reconstructed element stress are in MPa, and strain is dimensionless. The tensor columns follow the CPU writer’s order `(xx, yy, zz, xy, xz, yz)` with engineering shear strains. The stress uses the CUDA modulus and the same isotropic constitutive law at the element center.

## Run a native GPa file in place

~~~sh
OMP_NUM_THREADS=4 local/bin/parosol-cuda --format native existing-native.h5
~~~

This mode passes the existing file directly to CUDA without creating a copy or converting datasets. The solver writes `/Mesh` and `/Solution` into the same file using the native names and zero-based connectivity. Supply one path and use a native GPa file. The CPU and native inputs use the same modulus dataset name, so select the mode explicitly for a native input.

You can also call `local/bin/cudaparosol existing-native.h5` with the same OpenMP setting to run without Python.

## Architecture selection

With a numeric CMake architecture and no suffix, the build contains both native CUDA code and PTX. For example, `89-real` contains native code for 8.9 and `89-virtual` contains PTX. The driver can compile compatible PTX for the visible GPU. At startup the solver checks that the driver can load a solver kernel and reports an error if it cannot.

## Validation

Run the paired solve on one visible GPU to check results against the CPU executable from the same checkout.

~~~sh
python3 tests/compare_cuda_hdf5.py \
  --cpu build-cuda/parosol \
  --cuda build-cuda/src/parOsol_cuda/src/cudaparosol \
  --work-dir ./cuda-check
~~~

The test compares the mesh, reactions and seven solution datasets with the CPU result. It also checks native in-place output, including a five-slice model whose boundary nodes are translated beyond coordinate 32767. Use `--poisson-ratio 0.25` to select a different ratio.

## Source and license

The native CUDA port is based on Daniel Kellenberger’s [2014 CUDA ParOSol archive](https://bitbucket-archive.softwareheritage.org/projects/da/daniekel/cudaparosol.html), Mercurial revision `3553ee1668ac4a8ca0552ca393697c97fcf6066f`, and Cyril Flaig’s original ParOSol. The source follows the maintained IFB integration at revision `7760a1c2b193b70937e6f54c966b0d8639f423a3`, with subsequent shared fixes in the contribution history. The original notices and the native [license](../src/parOsol_cuda/LICENSE) accompany the source.
