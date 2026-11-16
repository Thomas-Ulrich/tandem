#include "PetscHierarchicalGreen.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <vector>

namespace tndm {
namespace {
struct Context {
    std::unique_ptr<HierarchicalMatrix> matrix;
    VecScatter scatter = nullptr;
    Vec global_slip = nullptr;
    std::vector<double> work_x, work_y;
    ~Context() {
        VecScatterDestroy(&scatter);
        VecDestroy(&global_slip);
    }
};

PetscErrorCode multiply(Mat shell, Vec x, Vec y) {
    Context* ctx;
    PetscScalar const* input;
    PetscScalar* output;
    PetscFunctionBeginUser;
    PetscCall(MatShellGetContext(shell, &ctx));
    PetscCall(VecScatterBegin(ctx->scatter, x, ctx->global_slip, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(ctx->scatter, x, ctx->global_slip, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecGetArrayRead(ctx->global_slip, &input));
    PetscCall(VecGetArray(y, &output));
#if !defined(PETSC_USE_COMPLEX) && defined(PETSC_USE_REAL_DOUBLE)
    ctx->matrix->multiply(input, output, ctx->work_x.data(), ctx->work_y.data());
#endif
    PetscCall(PetscLogFlops(2.0 * ctx->matrix->statistics().stored_scalars));
    PetscCall(VecRestoreArray(y, &output));
    PetscCall(VecRestoreArrayRead(ctx->global_slip, &input));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode destroy(Mat shell) {
    Context* ctx;
    PetscFunctionBeginUser;
    PetscCall(MatShellGetContext(shell, &ctx));
    delete ctx;
    PetscFunctionReturn(PETSC_SUCCESS);
}
} // namespace

PetscErrorCode compressGreenMatrix(Mat* matrix, Vec slip,
                                   std::vector<HierarchicalMatrix::Box> const& local_facets,
                                   PetscInt traction_block_size, PetscInt slip_block_size) {
    MPI_Comm comm = PetscObjectComm((PetscObject)*matrix);
    HierarchicalMatrix::Options options;
    PetscBool enabled = PETSC_FALSE;
    PetscReal tolerance = options.tolerance, eta = options.eta;
    PetscInt leaf = options.leaf_facets, rank_limit = options.max_rank;
    PetscInt m, n, M, N, lda;
    PetscFunctionBeginUser;
    PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-gf_hmatrix", &enabled, nullptr));
    if (!enabled) {
        PetscFunctionReturn(PETSC_SUCCESS);
    }
#if defined(PETSC_USE_COMPLEX) || !defined(PETSC_USE_REAL_DOUBLE)
    SETERRQ(comm, PETSC_ERR_SUP, "GF hierarchical prototype requires real double PETSc scalars");
#endif
    PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-gf_hmatrix_tol", &tolerance, nullptr));
    PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-gf_hmatrix_eta", &eta, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-gf_hmatrix_leaf", &leaf, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-gf_hmatrix_max_rank", &rank_limit, nullptr));
    PetscCheck(tolerance > 0 && tolerance < 1 && std::isfinite(eta) && eta > 0 && leaf > 0 &&
                   rank_limit > 0,
               comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid GF hierarchical options");
    options.tolerance = tolerance;
    options.eta = eta;
    options.leaf_facets = leaf;
    options.max_rank = rank_limit;
    PetscCall(MatGetLocalSize(*matrix, &m, &n));
    PetscCall(MatGetSize(*matrix, &M, &N));
    PetscCheck(traction_block_size > 0 && slip_block_size > 0, comm, PETSC_ERR_ARG_SIZ,
               "Invalid GF block sizes");
    PetscCheck(m == static_cast<PetscInt>(local_facets.size()) * traction_block_size &&
                   n == static_cast<PetscInt>(local_facets.size()) * slip_block_size,
               comm, PETSC_ERR_ARG_SIZ, "GF facet geometry does not match vector layouts");

    // Gather facet boxes in the same rank/facet order as PETSc's distributed columns.
    // Use doubles rather than communicating a C++ struct (which may contain padding).
    int size;
    PetscCallMPI(MPI_Comm_size(comm, &size));
    PetscMPIInt local_count;
    PetscCall(PetscMPIIntCast(6 * static_cast<PetscInt>(local_facets.size()), &local_count));
    std::vector<int> counts(size), displacements(size);
    PetscCallMPI(MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm));
    PetscInt total = 0;
    for (int p = 0; p < size; ++p) {
        PetscCall(PetscMPIIntCast(total, &displacements[p]));
        total += counts[p];
    }
    PetscMPIInt total_count;
    PetscCall(PetscMPIIntCast(total, &total_count));
    PetscCheck(total % 6 == 0 && total / 6 * slip_block_size == N &&
                   total / 6 * traction_block_size == M,
               comm, PETSC_ERR_ARG_SIZ, "Global GF geometry does not match matrix dimensions");
    std::vector<double> packed(local_count), gathered(total_count);
    for (std::size_t f = 0; f < local_facets.size(); ++f) {
        for (std::size_t d = 0; d < 3; ++d) {
            packed[6 * f + d] = local_facets[f].lo[d];
            packed[6 * f + 3 + d] = local_facets[f].hi[d];
        }
    }
    PetscCallMPI(MPI_Allgatherv(packed.data(), local_count, MPI_DOUBLE, gathered.data(),
                                counts.data(), displacements.data(), MPI_DOUBLE, comm));
    std::vector<HierarchicalMatrix::Box> global_facets(total / 6);
    for (std::size_t f = 0; f < global_facets.size(); ++f) {
        for (std::size_t d = 0; d < 3; ++d) {
            global_facets[f].lo[d] = gathered[6 * f + d];
            global_facets[f].hi[d] = gathered[6 * f + 3 + d];
        }
    }

    PetscLogDouble start, end;
    PetscCall(PetscTime(&start));
    auto ctx = std::make_unique<Context>();
    PetscScalar const* array;
    PetscCall(MatDenseGetLDA(*matrix, &lda));
    PetscCall(MatDenseGetArrayRead(*matrix, &array));
    int failed = 0, any_failed;
    try {
        ctx->matrix = std::make_unique<HierarchicalMatrix>(
            local_facets, global_facets, traction_block_size, slip_block_size, options,
            [array, lda](std::size_t i, std::size_t j) {
                return PetscRealPart(array[i + lda * j]);
            });
    } catch (std::exception const&) {
        failed = 1;
    }
    PetscCall(MatDenseRestoreArrayRead(*matrix, &array));
    PetscCallMPI(MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, comm));
    PetscCheck(!any_failed, comm, PETSC_ERR_LIB,
               "GF hierarchical construction failed (non-finite matrix or allocation failure)");
    ctx->work_x.resize(N);
    ctx->work_y.resize(m);
    PetscCall(VecScatterCreateToAll(slip, &ctx->scatter, &ctx->global_slip));
    Mat shell;
    PetscCall(MatCreateShell(comm, m, n, M, N, ctx.get(), &shell));
    PetscCall(MatShellSetOperation(shell, MATOP_MULT, (void (*)(void))multiply));
    PetscCall(MatShellSetOperation(shell, MATOP_DESTROY, (void (*)(void))destroy));
    ctx.release(); // Shell now owns the context.
    PetscCall(MatSetBlockSizes(shell, traction_block_size, 1));

    // Deterministic product checks before releasing the dense reference. These are
    // diagnostics, not a guarantee on relative error for every possible slip vector.
    Vec x, dense_y, compressed_y;
    PetscCall(MatCreateVecs(*matrix, &x, &dense_y));
    PetscCall(VecDuplicate(dense_y, &compressed_y));
    PetscInt begin, finish;
    PetscCall(VecGetOwnershipRange(x, &begin, &finish));
    PetscReal worst = 0;
    for (int trial = 0; trial < 3; ++trial) {
        PetscScalar* values;
        PetscCall(VecGetArray(x, &values));
        for (PetscInt i = begin; i < finish; ++i) {
            values[i - begin] = std::sin((i + 1.0) * (0.731 + trial * 0.317));
        }
        PetscCall(VecRestoreArray(x, &values));
        PetscCall(MatMult(*matrix, x, dense_y));
        PetscCall(MatMult(shell, x, compressed_y));
        PetscReal reference, error;
        PetscCall(VecNorm(dense_y, NORM_2, &reference));
        PetscCall(VecAXPY(compressed_y, -1.0, dense_y));
        PetscCall(VecNorm(compressed_y, NORM_2, &error));
        if (!std::isfinite(reference) || !std::isfinite(error)) {
            worst = std::numeric_limits<PetscReal>::infinity();
        } else {
            worst =
                std::max(worst, error / std::max(reference, std::numeric_limits<PetscReal>::min()));
        }
    }
    PetscInt repeats = 0;
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-gf_hmatrix_benchmark", &repeats, nullptr));
    if (repeats > 0) {
        double elapsed[2] = {};
        // Alternate order between rounds to reduce systematic warm-cache bias.
        for (int round = 0; round < 4; ++round) {
            for (int order = 0; order < 2; ++order) {
                int kernel = (round + order) % 2;
                PetscLogDouble t0, t1;
                PetscCallMPI(MPI_Barrier(comm));
                PetscCall(PetscTime(&t0));
                for (PetscInt i = 0; i < repeats; ++i) {
                    PetscCall(
                        MatMult(kernel ? shell : *matrix, x, kernel ? compressed_y : dense_y));
                }
                PetscCall(PetscTime(&t1));
                double local_ms = (t1 - t0) * 1000.0 / repeats, max_ms;
                PetscCallMPI(MPI_Allreduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, comm));
                elapsed[kernel] += max_ms / 4;
            }
        }
        PetscCall(PetscPrintf(
            comm, "GF H-matrix benchmark: dense_ms=%.6g compressed_ms=%.6g speedup=%.4g\n",
            elapsed[0], elapsed[1], elapsed[1] > 0 ? elapsed[0] / elapsed[1] : 0));
    }
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&dense_y));
    PetscCall(VecDestroy(&compressed_y));
    Context* context;
    PetscCall(MatShellGetContext(shell, &context));
    auto const& s = context->matrix->statistics();
    double local[] = {static_cast<double>(s.norm_squared), static_cast<double>(s.error_squared),
                      static_cast<double>(s.stored_scalars), static_cast<double>(s.dense_blocks),
                      static_cast<double>(s.low_rank_blocks)};
    double global[5];
    PetscCallMPI(MPI_Allreduce(local, global, 5, MPI_DOUBLE, MPI_SUM, comm));
    PetscInt local_rank = s.max_rank, max_rank;
    PetscCallMPI(MPI_Allreduce(&local_rank, &max_rank, 1, MPIU_INT, MPI_MAX, comm));
    PetscReal frobenius_error = global[0] > 0 ? std::sqrt(global[1] / global[0]) : 0;
    if (!std::isfinite(worst) || !std::isfinite(frobenius_error) ||
        frobenius_error > tolerance * (1 + 1e-12)) {
        PetscCall(MatDestroy(&shell));
        SETERRQ(comm, PETSC_ERR_NOT_CONVERGED, "GF hierarchical validation failed");
    }
    PetscCall(PetscTime(&end));
    PetscCall(PetscPrintf(
        comm,
        "GF H-matrix (CPU): dense_blocks=%.0f low_rank_blocks=%.0f max_rank=%" PetscInt_FMT "\n"
        "GF H-matrix: numerical_storage_GiB=%.6g dense_storage_GiB=%.6g compression=%.3gx\n"
        "GF H-matrix: relative_Frobenius_error=%.6g max_test_product_error=%.6g setup_s=%.6g\n",
        global[3], global[4], max_rank, global[2] * sizeof(double) / 1073741824.0,
        static_cast<double>(M) * N * sizeof(PetscScalar) / 1073741824.0,
        global[2] > 0 ? static_cast<double>(M) * N / global[2] : 1.0,
        static_cast<double>(frobenius_error), static_cast<double>(worst),
        static_cast<double>(end - start)));
    PetscCall(MatDestroy(matrix));
    *matrix = shell;
    PetscFunctionReturn(PETSC_SUCCESS);
}
} // namespace tndm
