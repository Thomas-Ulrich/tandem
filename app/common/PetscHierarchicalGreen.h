#ifndef PETSCHIERARCHICALGREEN_20261116_H
#define PETSCHIERARCHICALGREEN_20261116_H

#include "math/HierarchicalMatrix.h"
#include <petscmat.h>
#include <vector>

namespace tndm {

// Optional CPU compression after checkpoint loading/permutation. Replaces and destroys
// the dense matrix only after construction and distributed product checks succeed.
PetscErrorCode compressGreenMatrix(Mat* matrix, Vec slip,
                                   std::vector<HierarchicalMatrix::Box> const& local_facets,
                                   PetscInt traction_block_size, PetscInt slip_block_size);

} // namespace tndm
#endif
