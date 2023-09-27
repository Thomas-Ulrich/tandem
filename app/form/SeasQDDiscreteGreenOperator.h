#ifndef SEASQDDISCRETEGREENOPERATOR_20210907_H
#define SEASQDDISCRETEGREENOPERATOR_20210907_H

#include "common/PetscVector.h"
#include "form/AbstractAdapterOperator.h"
#include "form/AbstractFrictionOperator.h"
#include "form/FacetFunctionalFactory.h"
#include "form/SeasQDOperator.h"

#include <mpi.h>
#include <petscmat.h>
#include <petscvec.h>

#include <iostream>
#include <memory>
#include <utility>

namespace tndm {

class SeasQDDiscreteGreenOperator : public SeasQDOperator {
public:
    using base = SeasQDOperator;

    SeasQDDiscreteGreenOperator(std::unique_ptr<typename base::dg_t> dgop,
                                std::unique_ptr<AbstractAdapterOperator> adapter,
                                std::unique_ptr<AbstractFrictionOperator> friction,
                                bool matrix_free = false, MGConfig const& mg_config = MGConfig(), std::string prefix = "");
    ~SeasQDDiscreteGreenOperator();

    void set_boundary(std::unique_ptr<AbstractFacetFunctionalFactory> fun) override;

    inline void initial_condition(BlockVector& state) {
        base::friction().pre_init(state);

        update_traction(0.0, state);

        base::friction().init(0.0, base::traction_, state);
    }

    inline void rhs(double time, BlockVector const& state, BlockVector& result) {
        update_traction(time, state);

        base::friction().rhs(time, base::traction_, state, result);
    }

    void update_internal_state(double time, BlockVector const& state,
                               bool state_changed_since_last_rhs, bool require_traction,
                               bool require_displacement) override;

    std::tuple<std::string, std::string> get_checkpoint_filenames(void);
    double get_checkpoint_time_interval(void);
    void set_checkpoint_filenames(std::string, std::string);
    void set_checkpoint_time_interval(double);

protected:
    std::string gf_operator_filename_ = "gf_mat.bin";
    std::string gf_traction_filename_ = "gf_vec.bin";
    double checkpoint_every_nmins_ = 30.0;

    void update_traction(double time, BlockVector const& state);

private:
    void compute_discrete_greens_function();
    void compute_boundary_traction();
    void create_discrete_greens_function();
    void partial_assemble_discrete_greens_function();
    void write_discrete_greens_operator();
    void load_discrete_greens_operator();
    // all logic associated with matix craetion, loading / partial assembly is done here
    void get_discrete_greens_function();
    void write_discrete_greens_traction();
    void load_discrete_greens_traction();
    void get_boundary_traction();

    bool checkpoint_enabled_ = false;
    PetscInt current_gf_ = 0;
    PetscInt n_gf_ = 0;
    Mat G_ = nullptr;
    std::unique_ptr<PetscVector> S_;
    std::unique_ptr<PetscVector> t_boundary_;
};

} // namespace tndm

#endif // SEASQDDISCRETEGREENOPERATOR_20210907_H
