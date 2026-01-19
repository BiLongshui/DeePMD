// DeepPotHelper.h
#ifndef DEEP_POT_HELPER_H
#define DEEP_POT_HELPER_H

#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <iostream>

// DeepMD headers
#include "AtomMap.h"
#include "neighbor_list.h"
#include "common.h" 

// TensorFlow headers
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"

namespace LAMMPS_NS {

class DeepPotHelper {
public:
    DeepPotHelper();
    DeepPotHelper(const std::string& model, int gpu_rank = 0, const std::string& file_content = "");
    ~DeepPotHelper();

    void init(const std::string& model, int gpu_rank = 0, const std::string& file_content = "");

    void compute(double& ener,
                 std::vector<double>& force,
                 std::vector<double>& virial,
                 const std::vector<double>& coord,
                 const std::vector<int>& atype,
                 const std::vector<double>& box,
                 int nghost,
                 const deepmd::InputNlist& inlist,
                 int& ago,
                 const std::vector<double>& fparam = std::vector<double>(),
                 const std::vector<double>& aparam = std::vector<double>());

    // Getters
    double cutoff() const { assert(inited); return rcut; }
    int numb_types() const { assert(inited); return ntypes; }
    int numb_types_spin() const { assert(inited); return ntypes_spin; }
    int dim_fparam() const { assert(inited); return dfparam; }
    int dim_aparam() const { assert(inited); return daparam_; }
    void get_type_map(std::string& type_map);
    bool is_aparam_nall() const { assert(inited); return aparam_nall; }


private:
    // TensorFlow resources
    tensorflow::Session* session;
      int num_intra_nthreads, num_inter_nthreads;
    tensorflow::GraphDef* graph_def;


    // DeepMD resources
    deepmd::AtomMap atommap;
    deepmd::NeighborListData nlist_data;
    deepmd::InputNlist nlist;
    std::vector<int> sec_a;

    // Model attributes
    bool inited;
    double rcut;
    int dtype;
    double cell_size;
    int ntypes;
    int ntypes_spin;
    int dfparam;
    int daparam_;
    bool aparam_nall;
    std::string model_version;
    std::string model_type;

    bool init_nbor;

    // 内部辅助函数
    template <class VT>
    VT get_scalar(const std::string& name) const;

    template <typename VALUETYPE>
    void validate_fparam_aparam(const int& nframes,
                                const int& nloc,
                                const std::vector<VALUETYPE>& fparam,
                                const std::vector<VALUETYPE>& aparam) const;

    template <typename VALUETYPE>
    void tile_fparam_aparam(std::vector<VALUETYPE>& out_param,
                            const int& nframes,
                            const int& dparam,
                            const std::vector<VALUETYPE>& param) const;
    };
    
} // namespace LAMMPS_NS

#endif // DEEP_POT_HELPER_H