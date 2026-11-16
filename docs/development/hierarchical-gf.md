# Experimental hierarchical QDGreen operator

This CPU prototype compresses the existing double-precision GF matrix after its
assembly/checkpoint loading and repartitioning. The default path is unchanged.
No new library dependency is required.

Enable it through PETSc options, for example:

```sh
mpiexec -n 48 build/app/tandem Gofar32.toml --mg_coarse_level 1 --petsc \
    -options_file solver.cfg \
    -gf_hmatrix \
    -gf_hmatrix_tol 1e-6 \
    -gf_hmatrix_leaf 16 \
    -gf_hmatrix_max_rank 32 \
    -gf_hmatrix_eta 1 \
    -gf_hmatrix_benchmark 10
```

Use the same configuration, rank count, and solver options for a dense baseline,
omitting all `-gf_hmatrix*` options. Compare numerical storage, product timings,
complete time-stepping time, and fault histories. A useful tolerance sweep is
`1e-4`, `1e-6`, and `1e-8`, starting with a short run. No tolerance has yet been
validated for production earthquake-cycle simulations.

## Representation and accuracy

A binary spatial tree groups fault facets, keeping all their traction/slip
components together. Cluster bounding boxes use mesh facet vertices; for curved
facets these are clustering proxies rather than guaranteed physical bounds.
The acceptance test independently checks actual GF entries in either case.

Separated clusters satisfy
`max(diameter(row), diameter(column)) <= eta * distance(boxes)`.
Larger eta allows more blocks to be tried for compression.
Nearby terminal blocks remain dense. Separated blocks use fully pivoted cross
approximation, retaining factors only when they use fewer scalars than a dense
block and satisfy the requested relative Frobenius error. A failed compression
is subdivided, ending with exact dense storage if necessary. The maximum rank
is therefore a work/storage limit, not permission to exceed the error tolerance.

Each accepted compressed block meets `||B - U V^T||_F <= tol ||B||_F`.
The blocks partition the matrix, so the same bound holds for the complete
operator in exact arithmetic. This implies an absolute product error bound
`||(G-H)s||_2 <= tol ||G||_F ||s||_2`, not a relative product bound for every slip
vector. Startup also compares three deterministic products against the dense
matrix and prints their largest relative error.

Small blocks use a dense residual workspace (up to 262144 entries, except dense
terminal blocks). Large separated blocks stream residuals from the dense source,
so large compressible blocks remain intact without allocating a dense residual.
This deliberately conservative construction scans all block entries and can be
expensive. It is intended to establish achievable compression before replacing
construction with a faster algorithm.

## Diagnostics

- `numerical_storage_GiB`: dense leaf entries plus low-rank factors, summed over
  MPI ranks. This excludes tree/block metadata, vector workspace, MPI storage,
  and allocator overhead; it is not process RSS.
- `compression`: dense numerical storage divided by compressed numerical storage.
- `relative_Frobenius_error`: measured error from the stored factors.
- `max_test_product_error`: largest relative error from three startup products.
- `dense_ms`, `compressed_ms`, `speedup`: optional startup product benchmark,
  using the maximum rank time, four alternating rounds, and the requested number
  of repetitions per round. Setup timing includes checks and this benchmark.

## Current limits

- The dense matrix is still needed during setup, and the on-disk checkpoint
  remains dense. The dense matrix is released only after successful compression
  and validation. This reduces time-stepping storage; it does not yet enable a
  problem whose dense matrix cannot be loaded or assembled.
- Matrix products run on the CPU, including when the chosen vectors have a GPU
  backend (which can cause transfers). This is a prototype on main, separate
  from the CUDA dense-product branch.
- Original row ownership is retained, including ranks without fault elements.
  Slip is replicated using a precomputed PETSc scatter. Communication and work
  balancing are not optimized by this patch.
- No claim is made that the approximation preserves reciprocity, energy
  properties, adaptive time steps, or long-term event histories. These require
  physical validation in addition to the matrix error checks.
- PETSc must use real double scalars. The time integrator and friction law are
  unchanged.

The next construction stage would sample GF columns/operator applications and
write a compressed checkpoint directly, removing the dense setup/storage limit.

## Tests

```sh
cmake --build build --target test-hierarchical-matrix test-hierarchical-green -j
./build/test/test-hierarchical-matrix
mpiexec -n 4 ./build/test/test-hierarchical-green
```

The core tests reconstruct synthetic matrices to check the full Frobenius error,
permuted facet ordering, rectangular component blocks, dense fallback at a low
rank limit, streamed residual construction, zero blocks, empty row partitions,
and invalid inputs. The distributed test exercises the real PETSc shell,
repeated products, and ranks with no local fault facets.
