# Synopsis #

    parosol [options] input-file

# Description #

ParOSol is a fully-parallel micro-FE analysis code based on octrees to
solve linear elasticity problems. ParOSol solves the problem directly on
3D images, which can be obtained by CT-scans. It can be run on a desktop
machine as well as on a cluster using [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface).

# Using ParOSol #

If no options are given ParOSol solves up to a tolerance of 1e–6 and uses a
maximum of seven levels. An input file has to be provided. ParOSol writes
the result back into the input file once the solution is complete.

A sample command is given below:

    parosol --level 3 --tol 1e-4 simple.h5

This solves the elasticity equation on mesh given in the file
`simple.h5`. It uses maximal three multigrid levels and solves
with a relative residual of 1e–4.

If ParOSol was compiled using an MPI C++ compiler, you can use `mpirun` to let
ParOSol run in parallel:

    mpirun -n 4 parosol --level 3 --tol 1e-4 simple.h5

will run ParOSol using four threads.

## Eigenvalue estimation

`--eigen-max-iterations N` sets the **maximum** iterations for the adaptive
Chebyshev spectral estimate on each level (default 100, allowed 2–1000).
Every five steps, both the relative change of the largest Ritz value and
its relative Ritz residual must be below 1%, twice consecutively. Each
level stops independently; numerical Krylov termination can finish sooner.
The start vector is reproducible and varies by global node and component.
If the limit is reached without convergence, setup fails with a diagnostic;
increase the limit in that case. Normal convergence requires at least 15
steps, so a limit of 10 is generally insufficient.

The upper endpoint retains its 5% safety margin. This is still an estimate,
not a certified spectral bound; convergence targets the largest eigenvalue,
not the smallest. The iteration limit is separate from the polynomial degree.

For example:

```sh
mpirun -n 4 parosol --eigen-max-iterations 100 model.h5
```
