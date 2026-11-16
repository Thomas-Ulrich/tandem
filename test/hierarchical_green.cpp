#include "common/PetscHierarchicalGreen.h"

#include <cmath>
#include <vector>

// Standalone distributed PETSc integration test, including empty fault ranks.
static PetscErrorCode run() {
    MPI_Comm comm = PETSC_COMM_WORLD;
    int rank, size;
    PetscInt rb = 3, cb = 2, local_facets;
    PetscFunctionBeginUser;
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    PetscCallMPI(MPI_Comm_size(comm, &size));
    local_facets = rank < 2 ? 24 : 0;
    PetscInt offset = rank == 0 ? 0 : 24;
    std::vector<tndm::HierarchicalMatrix::Box> boxes(local_facets);
    for (PetscInt f = 0; f < local_facets; ++f) {
        double x = offset + f;
        boxes[f] = {{x - 0.2, 0, 0}, {x + 0.2, 0.1, 0}};
    }
    Mat dense, compressed;
    Vec x, y, reference;
    PetscCall(MatCreateDense(comm, rb * local_facets, cb * local_facets, PETSC_DECIDE, PETSC_DECIDE,
                             nullptr, &dense));
    PetscInt begin, end, M, N;
    PetscCall(MatGetOwnershipRange(dense, &begin, &end));
    PetscCall(MatGetSize(dense, &M, &N));
    for (PetscInt i = begin; i < end; ++i) {
        for (PetscInt j = 0; j < N; ++j) {
            double distance = static_cast<double>(i / rb - j / cb);
            PetscScalar value =
                (1.0 + 0.2 * (i % rb) - 0.1 * (j % cb)) / std::sqrt(0.01 + distance * distance);
            PetscCall(MatSetValue(dense, i, j, value, INSERT_VALUES));
        }
    }
    PetscCall(MatAssemblyBegin(dense, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(dense, MAT_FINAL_ASSEMBLY));
    PetscCall(MatCreateVecs(dense, &x, &y));
    PetscCall(VecDuplicate(y, &reference));
    PetscCall(VecSet(x, 1.0));
    PetscCall(MatMult(dense, x, reference));
    PetscCall(MatDuplicate(dense, MAT_COPY_VALUES, &compressed));
    PetscCall(PetscOptionsSetValue(nullptr, "-gf_hmatrix", "false"));
    Mat original = compressed;
    PetscCall(tndm::compressGreenMatrix(&compressed, x, boxes, rb, cb));
    PetscCheck(compressed == original, comm, PETSC_ERR_PLIB,
               "Disabled compression must preserve the original matrix");
    PetscCall(PetscOptionsSetValue(nullptr, "-gf_hmatrix", "true"));
    PetscCall(PetscOptionsSetValue(nullptr, "-gf_hmatrix_leaf", "4"));
    PetscCall(PetscOptionsSetValue(nullptr, "-gf_hmatrix_tol", "1e-6"));
    PetscCall(tndm::compressGreenMatrix(&compressed, x, boxes, rb, cb));
    for (int repeat = 0; repeat < 3; ++repeat) {
        PetscCall(MatMult(compressed, x, y));
        PetscReal norm, error;
        PetscCall(VecNorm(reference, NORM_2, &norm));
        PetscCall(VecAXPY(y, -1.0, reference));
        PetscCall(VecNorm(y, NORM_2, &error));
        PetscCheck(error / norm < 1e-5, comm, PETSC_ERR_PLIB,
                   "Distributed hierarchical product error %g", static_cast<double>(error / norm));
    }
    PetscCall(MatDestroy(&compressed));
    PetscCall(MatDestroy(&dense));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&y));
    PetscCall(VecDestroy(&reference));
    PetscCall(PetscPrintf(comm, "Hierarchical GF integration test passed on %d rank(s)\n", size));
    PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char** argv) {
    PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr));
    PetscCall(run());
    return PetscFinalize();
}
