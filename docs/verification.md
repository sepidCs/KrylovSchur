# Numerical checks

## Environment and protocol

Windows 11, CPython 3.11.9, Intel Core i5-8365U
(4 cores, 8 logical processors). Numerical work uses one thread.
The dependency versions and thread-pool configuration are recorded in
`results/metrics.json`; direct versions are pinned in `requirements.txt`.

`python -m experiments.compare --suite all --repeats 5` contains 15 cases and 52 method/case
measurements. Each method has one warm-up followed by five timed runs.
The recorded timing run has no concurrent project tests or calculations.
The same arrays, rank, and RNG state are used for each method in a case.

Data use `np.random.RandomState(0)`. Before each solver run, NumPy and
PyTorch seeds are set to 1. Accuracy uses the final run's factors.
Input creation, RNG resets, imports, reconstruction, norms, and plots are
excluded from timing. Gram formation, tensor FFTs, factor inverse FFTs,
and tensor/NumPy conversions inside the solver are included.

Matrix inputs are float64. The Krylov-Schur basis and projected matrices
are complex128, including for real matrices. Tensor Fourier slices are
complex128 for both the dense and iterative tensor methods.
Thus timings describe these implementations and their conversion overheads.

## Accuracy checks

`tests/test_numerics.py` contains 11 passing test methods covering:

- Tall, wide, complex, zero, and rank-deficient matrices.
- Small singular values and orthonormal factors.
- Krylov convergence counts and an invariant-subspace relation.
- A visible nonconvergence warning when the iteration budget is zero.
- Complex circular tensor convolution and Fourier-domain tensor adjoints.
- Tensor truncation errors against per-slice dense SVD.
- Full-rank PyTorch tensor reconstruction.

Dense-SVD residual comparisons use `rtol=1e-9, atol=1e-10`.
Convolution/adjoint checks use `rtol=1e-12, atol=1e-12`.
Tolerance-based agreement is not byte-identical numerical equivalence.

All 15 benchmark Krylov-Schur residuals agree with the dense SVD reference.
The largest observed `abs(error_over_dense - 1)` is approximately `3.33e-15`.
No solver warnings or nonfinite factors were recorded in the benchmark.

## Performance findings

The implementation has no measured speed advantage in this suite.
Krylov-Schur is slower than dense SVD and PyTorch in every measured case.
At `1000x100x10`, rank 30, median times are 114.591 ms for PyTorch,
232.226 ms for dense t-SVD, and 342.836 ms for Krylov-Schur t-SVD.
Corresponding relative errors are 39.4232%, 38.1137%, and 38.1137%.

PyTorch uses two power iterations with `q=k`; scikit-learn uses its
five-iteration default; Krylov-Schur uses tolerance `2e-16`, the stated
restart size, and a maximum of 100 or 150 iterations.
The methods use different stopping rules. The error difference describes
these settings; it does not show which method is faster at the same error.

The operation-count model includes dense Gram formation `O(q p^2)`,
Krylov expansion/orthogonalization, Schur/restart work, and projected SVD.
It does not reduce to an eigensolver-only `O(p^3)` claim.
Auxiliary storage includes `O(p^2 + p r + q k)`, excluding input/output.
The tensor implementation processes all Fourier slices; it does not exploit
conjugate symmetry to skip half the slices. Peak memory was not measured.

## Reproduction

From the repository root:

```powershell
./.venv/Scripts/python.exe -m pip check
./.venv/Scripts/python.exe -m unittest discover -s tests -v
./.venv/Scripts/python.exe -m experiments.compare --suite all --repeats 5
```

The recorded measurements use the project environment and random test data.
Direct dependencies are pinned in `requirements.txt`; their dependencies
are not fully locked. Other operating systems, other Python versions, GPUs,
and application datasets are outside the tested setup.

## Interpretation

The tests compare results with full dense SVD, direct circular convolution,
and matrices with known eigenvalues. They cover the listed cases rather
than every possible input. The tests and comparison script use the same
calculation modules.

The Gram formulation squares the condition number. Difficult spectra,
nonsymmetric matrices, arbitrary precisions, and sparse or implicit operators
need broader tests before a general-purpose library contribution.
