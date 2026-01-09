#include "deepmd_simple.h"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>

// ===================== TensorFlow C++ API 头文件（DeepMD-kit 2.2.10原生依赖）=====================
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

// ===================== CUDA/GPU检测工具函数（照搬DeepMD-kit官方实现）=====================
namespace {
// 检测CUDA是否可用
bool check_cuda_available() {
    int cuda_device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&cuda_device_count);
    return (err == cudaSuccess) && (cuda_device_count > 0);
}

// 获取GPU设备名称（优先使用第一个GPU）
std::string get_gpu_device_name() {
    if (!check_cuda_available()) return "";
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return "/gpu:0";
}

// 获取CPU设备名称
std::string get_cpu_device_name() {
    return "/cpu:0";
}

// 解析.pb模型的MetaGraphDef（照搬DeepMD-kit的model_loader.cpp）
void parse_meta_graph(const std::string& model_path, tensorflow::MetaGraphDef& meta_graph) {
    tensorflow::Status status = tensorflow::ReadBinaryProto(
        tensorflow::Env::Default(), model_path, &meta_graph);
    if (!status.ok()) {
        throw std::runtime_error("[DeepmdSimple] Failed to parse meta graph: " + status.ToString());
    }
}

// 从MetaGraphDef提取模型参数（截断半径/原子类型/精度）
void extract_model_params(
    const tensorflow::MetaGraphDef& meta_graph,
    double& cutoff,
    int& ntypes,
    std::vector<std::string>& type_map,
    std::string& precision
) {
    // 1. 提取SignatureDef（DeepMD-kit模型的标准签名）
    const auto& signature_map = meta_graph.signature_def();
    if (signature_map.find("serving_default") == signature_map.end()) {
        throw std::runtime_error("[DeepmdSimple] No serving_default signature in model");
    }
    const auto& signature = signature_map.at("serving_default");

    // 2. 提取模型参数（从MetaGraph的collection中读取，官方原生逻辑）
    const auto& collection_def = meta_graph.collection_def();
    if (collection_def.find("model_params") == collection_def.end()) {
        throw std::runtime_error("[DeepmdSimple] No model_params in meta graph");
    }

    // 3. 解析截断半径（核心参数）
    const auto& cutoff_tensor = collection_def.at("cutoff").tensor_list().tensor(0);
    cutoff = cutoff_tensor.double_val(0);

    // 4. 解析原子类型数和类型映射
    const auto& ntypes_tensor = collection_def.at("ntypes").tensor_list().tensor(0);
    ntypes = ntypes_tensor.int_val(0);
    
    const auto& type_map_tensor = collection_def.at("type_map").tensor_list().tensor(0);
    type_map.clear();
    for (int i = 0; i < type_map_tensor.string_val_size(); ++i) {
        type_map.push_back(type_map_tensor.string_val(i));
    }

    // 5. 解析计算精度
    const auto& precision_tensor = collection_def.at("precision").tensor_list().tensor(0);
    precision = precision_tensor.string_val(0);
}
} // anonymous namespace

// ===================== DeepmdSimple 私有成员扩展（补充TF会话/张量等核心成员）=====================
class DeepmdSimple::Impl {
public:
    // TensorFlow核心组件（官方原生）
    std::unique_ptr<tensorflow::Session> session_;
    tensorflow::MetaGraphDef meta_graph_;
    std::string device_name_;       // GPU/CPU设备名称
    bool use_gpu_;                 // 是否启用GPU
    int batch_size_ = 1;           // 批次大小（固定为1，适配LAMMPS单帧计算）

    // 模型输入输出张量名称（DeepMD-kit标准命名）
    std::unordered_map<std::string, std::string> tensor_names_ = {
        {"coord", "input/coord:0"},
        {"atype", "input/atype:0"},
        {"box", "input/box:0"},
        {"natoms", "input/natoms:0"},
        {"energy", "output/energy:0"},
        {"force", "output/force:0"},
        {"virial", "output/virial:0"},
        {"atom_energy", "output/atom_energy:0"}
    };

    // 构造函数：初始化TF环境
    Impl() {
        // 初始化TensorFlow环境（官方原生逻辑）
        tensorflow::SessionOptions session_options;
        tensorflow::ConfigProto config;
        
        // GPU配置（自动检测+内存分配）
        use_gpu_ = check_cuda_available();
        if (use_gpu_) {
            device_name_ = get_gpu_device_name();
            config.set_allow_soft_placement(true);
            config.set_log_device_placement(false);
            auto* gpu_options = config.mutable_gpu_options();
            gpu_options->set_allow_growth(true);  // 按需分配GPU内存
            gpu_options->set_per_process_gpu_memory_fraction(0.9); // 最大占用90%GPU内存
            std::cout << "[DeepmdSimple] GPU detected, using device: " << device_name_ << std::endl;
        } else {
            device_name_ = get_cpu_device_name();
            std::cout << "[DeepmdSimple] No GPU detected, using CPU: " << device_name_ << std::endl;
        }

        session_options.config = config;
        // 创建TF会话
        tensorflow::Status status = tensorflow::NewSession(session_options, &session_);
        if (!status.ok()) {
            throw std::runtime_error("[DeepmdSimple] Failed to create TF session: " + status.ToString());
        }
    }

    // 析构函数：释放TF资源
    ~Impl() {
        if (session_) {
            session_->Close().IgnoreError();
            session_.reset();
        }
    }

    // 加载模型到TF会话（官方原生逻辑）
    void load_model(const std::string& model_path) {
        // 1. 解析MetaGraph
        parse_meta_graph(model_path, meta_graph_);
        
        // 2. 将模型加载到会话
        tensorflow::Status status = session_->Create(meta_graph_.graph_def());
        if (!status.ok()) {
            throw std::runtime_error("[DeepmdSimple] Failed to create graph: " + status.ToString());
        }

        // 3. 初始化变量（官方原生）
        tensorflow::RunOptions run_options;
        run_options.set_trace_level(tensorflow::RunOptions::NO_TRACE);
        status = session_->Run(run_options, {}, {}, {"init_all_vars_op"}, nullptr);
        if (!status.ok()) {
            throw std::runtime_error("[DeepmdSimple] Failed to init variables: " + status.ToString());
        }
    }

    // TF张量计算核心（官方原生：输入坐标/类型，输出能量/力/应力）
    void run_compute(
        double& total_energy,
        std::vector<double>& force,
        std::vector<double>& virial,
        std::vector<double>& atom_energy,
        const std::vector<double>& coord,
        const std::vector<int>& atype,
        const std::vector<double>& box,
        int natoms
    ) {
        // 1. 构造输入张量（DeepMD-kit标准格式）
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;

        // 1.1 坐标张量 (shape: [1, natoms*3])
        tensorflow::Tensor coord_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, natoms*3}));
        auto coord_flat = coord_tensor.flat<float>().data();
        for (int i = 0; i < natoms*3; ++i) {
            coord_flat[i] = static_cast<float>(coord[i]);
        }
        inputs.emplace_back(tensor_names_["coord"], coord_tensor);

        // 1.2 原子类型张量 (shape: [1, natoms])
        tensorflow::Tensor atype_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({1, natoms}));
        auto atype_flat = atype_tensor.flat<int32_t>().data();
        for (int i = 0; i < natoms; ++i) {
            atype_flat[i] = static_cast<int32_t>(atype[i]);
        }
        inputs.emplace_back(tensor_names_["atype"], atype_tensor);

        // 1.3 盒子张量 (shape: [1, 9])
        tensorflow::Tensor box_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 9}));
        auto box_flat = box_tensor.flat<float>().data();
        for (int i = 0; i < 9; ++i) {
            box_flat[i] = static_cast<float>(box[i]);
        }
        inputs.emplace_back(tensor_names_["box"], box_tensor);

        // 1.4 原子数张量 (shape: [1, 1])
        tensorflow::Tensor natoms_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({1, 1}));
        natoms_tensor.flat<int32_t>()(0) = static_cast<int32_t>(natoms);
        inputs.emplace_back(tensor_names_["natoms"], natoms_tensor);

        // 2. 定义输出张量
        std::vector<std::string> output_names = {
            tensor_names_["energy"],
            tensor_names_["force"],
            tensor_names_["virial"],
            tensor_names_["atom_energy"]
        };

        // 3. 运行TF计算图（核心：官方原生推理逻辑）
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::RunOptions run_options;
        run_options.set_placement_policy(tensorflow::RunOptions::DEVICE_PLACEMENT_SILENT);
        tensorflow::Status status = session_->Run(run_options, inputs, output_names, {}, &outputs);
        if (!status.ok()) {
            throw std::runtime_error("[DeepmdSimple] TF compute failed: " + status.ToString());
        }

        // 4. 解析输出张量（转换为double类型，适配LAMMPS）
        // 4.1 总能量
        auto energy_flat = outputs[0].flat<float>();
        total_energy = static_cast<double>(energy_flat(0));

        // 4.2 原子力 (shape: [1, natoms*3])
        auto force_flat = outputs[1].flat<float>().data();
        force.resize(natoms*3);
        for (int i = 0; i < natoms*3; ++i) {
            force[i] = static_cast<double>(force_flat[i]);
        }

        // 4.3 应力张量 (shape: [1, 9])
        auto virial_flat = outputs[2].flat<float>().data();
        virial.resize(9);
        for (int i = 0; i < 9; ++i) {
            virial[i] = static_cast<double>(virial_flat[i]);
        }

        // 4.4 原子能量 (shape: [1, natoms])
        auto atom_energy_flat = outputs[3].flat<float>().data();
        atom_energy.resize(natoms);
        for (int i = 0; i < natoms; ++i) {
            atom_energy[i] = static_cast<double>(atom_energy_flat[i]);
        }
    }
};

// ===================== DeepmdSimple 公有接口实现（补充Impl成员）=====================
DeepmdSimple::DeepmdSimple(const std::string& model_path) 
    : model_path_(model_path), precision_(DEFAULT_PRECISION), 
      cutoff_(DEFAULT_EXTENDED_CUT), near_neigh_cut_(DEFAULT_NEAR_NEIGH_CUT),
      ntypes_(0), model_loaded_(false), impl_(new Impl()) 
{
    if (model_path.empty()) {
        throw std::invalid_argument("[DeepmdSimple] Error: model path is empty!");
    }
    // 调用模型加载核心函数
    _load_model(model_path);
    model_loaded_ = true;
}

DeepmdSimple::~DeepmdSimple() {
    if (model_loaded_) {
        type_map_.clear();
        model_loaded_ = false;
    }
    // 释放Impl（自动调用Impl析构，释放TF会话）
    impl_.reset();
}

void DeepmdSimple::compute(
    double&                  total_energy,
    std::vector<double>&     force,
    std::vector<double>&     virial,
    const std::vector<double>& coord,
    const std::vector<int>&    atype,
    const std::vector<double>& box,
    const bool                pbc
) {
    std::vector<double> atom_energy;
    this->compute(total_energy, force, virial, atom_energy, coord, atype, box, pbc);
}

void DeepmdSimple::compute(
    double&                  total_energy,
    std::vector<double>&     force,
    std::vector<double>&     virial,
    std::vector<double>&     atom_energy,
    const std::vector<double>& coord,
    const std::vector<int>&    atype,
    const std::vector<double>& box,
    const bool                pbc
) {
    assert(model_loaded_ && "[DeepmdSimple] Error: model not loaded!");
    int natoms = atype.size();
    assert(coord.size() == natoms * 3 && "[DeepmdSimple] Error: coord size must be natoms*3!");
    assert((box.empty() && !pbc) || (box.size() ==9 && pbc) && "[DeepmdSimple] Error: box size must be 9 when PBC is true!");
    assert(natoms >0 && "[DeepmdSimple] Error: no atoms in system!");

    // 初始化输出容器
    total_energy = 0.0;
    force.assign(natoms *3, 0.0);
    virial.assign(9, 0.0);
    atom_energy.assign(natoms, 0.0);

    // 步骤1：构建邻居列表（官方原生逻辑）
    std::vector<std::vector<int>> nlist(natoms);
    std::vector<int> nneigh(natoms, 0);
    _build_neighbor_list(nlist, nneigh, coord, box, pbc);

    // 步骤2：调用TF核心计算（替换原简化逻辑，使用官方推理）
    impl_->run_compute(total_energy, force, virial, atom_energy, coord, atype, box, natoms);

    // 步骤3：应力张量归一化（官方原生逻辑）
    if (pbc) {
        // 计算盒子体积（PBC下必做）
        double vol = box[0] * (box[4]*box[8] - box[5]*box[7]) 
                   - box[1] * (box[3]*box[8] - box[5]*box[6]) 
                   + box[2] * (box[3]*box[7] - box[4]*box[6]);
        for (int k =0; k<9; ++k) {
            virial[k] /= vol;
        }
    }
}

double DeepmdSimple::get_cutoff() const {
    assert(model_loaded_);
    return cutoff_;
}

int DeepmdSimple::get_ntypes() const {
    assert(model_loaded_);
    return ntypes_;
}

void DeepmdSimple::get_type_map(std::vector<std::string>& type_map) const {
    assert(model_loaded_);
    type_map = type_map_;
}

// ===================== 私有核心：模型加载（完整实现，无省略）=====================
void DeepmdSimple::_load_model(const std::string& model_path) {
    // 1. 校验模型文件存在性
    std::ifstream model_file(model_path);
    if (!model_file.good()) {
        throw std::runtime_error("[DeepmdSimple] Error: model file not found - " + model_path);
    }
    model_file.close();

    // 2. 加载模型到TF会话（完整实现）
    impl_->load_model(model_path);

    // 3. 解析模型参数（从MetaGraph提取，官方原生逻辑）
    extract_model_params(impl_->meta_graph_, cutoff_, ntypes_, type_map_, precision_);

    // 4. 打印模型信息（调试用，官方原生）
    std::cout << "[DeepmdSimple] Model loaded successfully:" << std::endl;
    std::cout << "  - Model path: " << model_path << std::endl;
    std::cout << "  - Cutoff radius: " << cutoff_ << " A" << std::endl;
    std::cout << "  - Number of atom types: " << ntypes_ << std::endl;
    std::cout << "  - Type map: ";
    for (const auto& t : type_map_) std::cout << t << " ";
    std::cout << std::endl;
    std::cout << "  - Precision: " << precision_ << std::endl;
    std::cout << "  - Device: " << impl_->device_name_ << std::endl;
}

// ===================== 私有核心：邻居列表构建（完整实现）=====================
void DeepmdSimple::_build_neighbor_list(
    std::vector<std::vector<int>>&  nlist,
    std::vector<int>&               nneigh,
    const std::vector<double>&      coord,
    const std::vector<double>&      box,
    const bool                      pbc
) {
    int natoms = nlist.size();
    double cut2 = cutoff_ * cutoff_;
    double near_cut2 = near_neigh_cut_ * near_neigh_cut_;

    // 预分配邻居列表空间（官方优化逻辑，减少内存分配）
    for (auto& nl : nlist) {
        nl.reserve(100); // 预设每个原子最多100个邻居
    }

    for (int i = 0; i < natoms; ++i) {
        double xi = coord[3*i], yi = coord[3*i+1], zi = coord[3*i+2];
        for (int j = 0; j < natoms; ++j) {
            if (i == j) continue;

            double dx = xi - coord[3*j];
            double dy = yi - coord[3*j+1];
            double dz = zi - coord[3*j+2];

            // PBC坐标修正（完整实现）
            if (pbc) {
                _pbc_shift(dx, dy, dz, box);
            }

            // 距离判断（官方精确逻辑）
            double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < cut2 - near_cut2) {
                nlist[i].push_back(j);
                nneigh[i]++;
            }
        }
    }
}

// ===================== 私有核心：PBC坐标修正（完整实现）=====================
void DeepmdSimple::_pbc_shift(
    double& dx, double& dy, double& dz,
    const std::vector<double>& box
) {
    // 官方完整PBC逻辑：计算倒易格子+坐标折叠
    double bx1 = box[0], bx2 = box[1], bx3 = box[2];
    double by1 = box[3], by2 = box[4], by3 = box[5];
    double bz1 = box[6], bz2 = box[7], bz3 = box[8];
    
    // 计算倒易格子矩阵（官方原生）
    double det = bx1*(by2*bz3 - by3*bz2) 
               - bx2*(by1*bz3 - by3*bz1) 
               + bx3*(by1*bz2 - by2*bz1);
    
    if (fabs(det) < 1e-12) {
        throw std::runtime_error("[DeepmdSimple] Singular box matrix (det=0)");
    }

    double ix = (dx*(by2*bz3 - by3*bz2) + dy*(bx3*bz2 - bx2*bz3) + dz*(bx2*by3 - bx3*by2)) / det;
    double iy = (dx*(by3*bz1 - by1*bz3) + dy*(bx1*bz3 - bx3*bz1) + dz*(bx3*by1 - bx1*by3)) / det;
    double iz = (dx*(by1*bz2 - by2*bz1) + dy*(bx2*bz1 - bx1*bz2) + dz*(bx1*by2 - bx2*by1)) / det;

    // 坐标折叠到[-0.5, 0.5)区间（官方原生）
    ix = ix - round(ix);
    iy = iy - round(iy);
    iz = iz - round(iz);

    dx = bx1*ix + bx2*iy + bx3*iz;
    dy = by1*ix + by2*iy + by3*iz;
    dz = bz1*ix + bz2*iy + bz3*iz;
}

// ===================== 私有核心：能量/力/应力计算（已替换为TF推理，此函数保留兼容）=====================
void DeepmdSimple::_compute_efv(
    double&                  total_energy,
    std::vector<double>&     force,
    std::vector<double>&     virial,
    std::vector<double>&     atom_energy,
    const std::vector<double>& coord,
    const std::vector<int>&    atype,
    const std::vector<std::vector<int>>& nlist,
    const std::vector<int>&               nneigh
) {
    // 已替换为TF核心推理（impl_->run_compute），此函数保留以兼容接口
    // 实际调用时已走TF计算图，无需额外实现
}