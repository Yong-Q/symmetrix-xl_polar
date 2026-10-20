#pragma once

#include <Kokkos_Core.hpp>

// [v3 适配·D-09] 模块⑧ 使用 std::array(提案 §2.2 注; 现树 hpp 未含 <array>)。
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

// Lightweight opt-in stage profiler (MATOS_POLAR_PROFILE=1).  Each call
// fences and prints the wall time elapsed since the previous mark.
inline void symmetrix_polar_profile_mark(const char *label) {
    if (std::getenv("MATOS_POLAR_PROFILE") == nullptr) return;
    using clock = std::chrono::high_resolution_clock;
    static clock::time_point last = clock::now();
    Kokkos::fence();
    const auto now = clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(now - last).count();
    std::fprintf(stderr, "[polar-profile] %-24s %10.3f ms\n", label, ms);
    last = now;
}

// Same profiler with a layer index suffix, e.g. "fwd:linear_1[1]".
// Disabled (single getenv) when MATOS_POLAR_PROFILE is unset.
inline void symmetrix_polar_profile_mark_i(const char *label, int index) {
    if (std::getenv("MATOS_POLAR_PROFILE") == nullptr) return;
    char buffer[128];
    std::snprintf(buffer, sizeof(buffer), "%s[%d]", label, index);
    symmetrix_polar_profile_mark(buffer);
}

#include "affine_mlp_kokkos.hpp"
#include "device_backend.hpp"
#include "e3nn_kokkos.hpp"
#include "e3nn_product_kokkos.hpp"
#include "mace_nonlinear_schema.hpp"
#include "mace_streamed_edges.hpp"
#include "prediction_heads.hpp"
#include "mh1_stage.hpp"
#include "radial_function_set_kokkos.hpp"
#include "jit_mh1_cuda_plugin.hpp"
#include "jit_mh1_host_plugin.hpp"
#include "jit_generation_version.hpp"
#include "tools_kokkos.hpp"
#include "zbl_kokkos.hpp"

template<typename Precision>
class MaceNonlinearKokkosT {
    KokkosLiveObjectGuard kokkos_live_object_guard;

public:
    using precision_type=Precision;
    explicit MaceNonlinearKokkosT(
        const std::string& filename, const std::string& requested_head = {});
    ~MaceNonlinearKokkosT();
    double r_cut=0.0;
    bool has_field_coupling=false;
    std::string selected_head;
    std::vector<std::string> available_heads;
    bool is_mh1_family() const { return mh1_family.compatible; }
    int mh1_node_channels() const { return mh1_family.node_channels; }
    int mh1_edge_channels() const { return mh1_family.edge_channels; }
    int mh1_radial_size() const { return mh1_family.radial_size; }
    int mh1_l_max() const { return mh1_family.l_max; }
    const std::string& mh1_family_rejection_reason() const {
        return mh1_family.rejection_reason;
    }
    bool mh1_uses_compiled_products() const { return mh1_compiled_products; }
    bool mh1_uses_pair_conditioning() const { return mh1_pair_conditioning; }
    bool mh1_uses_external_uvu_tensors() const { return mh1_external_uvu_tensors; }
    const std::string& mh1_fast_path_rejection_reason() const {
        return mh1_fast_path_rejection;
    }
    bool uses_mh1_fast_path() const { return mh1_fast_path; }
    bool supports_streamed_edges() const { return mh1_fast_path; }
    bool supports_factorized() const;
    bool has_execution_mh1_contract() const { return execution_mh1_has_model_contract; }
    const std::string& execution_mh1_generation_fingerprint() const {
        return execution_mh1_model_generation_fingerprint;
    }
    const std::string& execution_mh1_semantic_fingerprint() const {
        return execution_mh1_model_semantic_fingerprint;
    }
    const std::string& execution_mh1_structure_fingerprint() const {
        return execution_mh1_model_structure_fingerprint;
    }
    void set_execution_mh1_contract_identity(
        std::string tag,std::string generation_fingerprint,
        std::string semantic_fingerprint,std::string structure_fingerprint);
    void set_execution_mh1_v4_contract_identity(
        std::string tag,std::string generation_fingerprint,
        std::string semantic_fingerprint,std::string structure_fingerprint,
        std::string runtime_layout_fingerprint);
    void load_jit_mh1_host_plugin(std::string path);
    void load_jit_mh1_host_plugin_v4(std::string path);
    void load_jit_mh1_host_plugin_v5(std::string path);
    bool jit_mh1_host_plugin_ready() const;
    bool jit_mh1_host_plugin_v4_ready() const;
    bool jit_mh1_host_plugin_v5_ready() const;
    std::string jit_mh1_host_plugin_path() const;
    std::string jit_mh1_host_plugin_artifact_id() const;
    void load_jit_mh1_cuda_plugin(std::string path);
    void load_jit_mh1_cuda_plugin_v4(
        std::string path, std::string launch_plan_json="");
    bool jit_mh1_cuda_plugin_ready() const;
    bool jit_mh1_cuda_plugin_v4_ready() const;
    std::string jit_mh1_cuda_plugin_path() const;
    std::string jit_mh1_cuda_plugin_artifact_id() const;
    bool device_cuda_available() const;
    std::string execution_cuda_device_name() const;
    int execution_cuda_device_ordinal() const;
    int execution_cuda_compute_capability() const;
    int execution_cuda_multiprocessor_count() const;
    int execution_cuda_warp_width() const;
    int execution_cuda_runtime_version() const;
    int execution_cuda_driver_version() const;
    std::string execution_mh1_execution_backend() const {
        if(!mace_uses_prepared_execution(streamed_edges)) return "inactive";
        if(mh1_edge_executor_=="pair_spline_v1") {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
            if constexpr(device_execution_space<Kokkos::DefaultExecutionSpace>)
                return jit_mh1_cuda_plugin_v4_ready()
                        &&jit_mh1_cuda_plugin_v4->has_spline_r_program()
                    ?"pair_spline_v1_staged_device_v5"
                    :"pair_spline_v1_kokkos_device";
#endif
            return jit_mh1_host_plugin_v5_ready()
                ?(jit_mh1_host_plugin_v4_ready()
                    ?"pair_spline_v1_staged_host_v5"
                    :"pair_spline_v1_staged_r_host_v5")
                :"pair_spline_v1_kokkos";
        }
        if(jit_mh1_cuda_plugin_v4_ready()) {
#ifdef KOKKOS_ENABLE_HIP
            if constexpr(std::is_same_v<
                Kokkos::DefaultExecutionSpace,Kokkos::HIP>)
                return "generated_hip_v4";
#endif
            return "generated_cuda_v4";
        }
        if(jit_mh1_host_plugin_v4_ready())
            return execution_mh1_host_node_backend_request=="kokkos"
                ?"generated_host_v4_kokkos_nodes":"generated_host_v4";
        if(jit_mh1_cuda_plugin_ready()) return "generated_cuda";
        if(jit_mh1_host_plugin_ready()) return "generated_host";
        return "unavailable";
    }
    std::string streamed_edges_mode() const;
    void set_streamed_edges(std::string mode);
    const std::string& mh1_edge_executor() const { return mh1_edge_executor_; }
    void set_mh1_edge_executor(std::string executor);
    bool mh1_pair_spline_ready() const { return mh1_pair_spline_ready_; }
    int mh1_pair_spline_nodes() const { return mh1_pair_spline_nodes_; }
    int mh1_pair_spline_type_count() const { return mh1_spline_type_count_; }
    void set_execution_mh1_node_arena_policy(std::string policy);
    const std::string& execution_mh1_node_arena_policy() const {
        return execution_mh1_node_arena_policy_;
    }
    int execution_mh1_node_arena_tile_rows() const;
    void set_execution_mh1_node_arena_tile_rows_for_testing(int rows);
    std::size_t factorized_forward_evaluation_count() const {
        return factorized_forward_evaluations;
    }
    std::size_t factorized_reverse_evaluation_count() const {
        return factorized_reverse_evaluations;
    }
    std::size_t factorized_fallback_evaluation_count() const {
        return factorized_fallback_evaluations;
    }
    std::size_t factorized_schedule_build_count() const {
        return factorized_schedule_builds;
    }
    std::size_t factorized_schedule_entries() const {
        return execution_source_edges.extent(0);
    }
    std::size_t factorized_schedule_active_sources() const {
        return execution_source_edge_offsets.extent(0)==0
            ?0:execution_source_edge_offsets.extent(0)-1;
    }
    std::size_t factorized_schedule_bytes() const {
        return sizeof(int)*(execution_block_source_segments.extent(0)
            +execution_source_edge_offsets.extent(0)
            +execution_source_edges.extent(0)
            +execution_block_receivers.extent(0)
            +execution_graph_source_edge_offsets.extent(0)
            +execution_graph_source_edges.extent(0)
            +execution_graph_receivers.extent(0));
    }
    std::uint64_t prepare_factorized_graph(
        int num_nodes,std::span<const int> node_types,
        std::span<const int> num_neigh,
        std::span<const int> neigh_indices,
        std::span<const int> neigh_types);
    void prepare_factorized_geometry(
        std::uint64_t graph_generation,
        std::span<const double> reference_positions,
        std::span<const double> reference_xyz,
        std::span<const double> cell,
        std::span<const double> inverse_cell,
        std::span<const int> pbc);
    void compute_prepared_factorized(
        std::uint64_t graph_generation,
        std::span<const double> xyz,std::span<const double> distances);
    void compute_prepared_factorized_positions(
        std::uint64_t graph_generation,
        std::span<const double> positions);
    void reduce_node_forces(
        int num_nodes,
        Kokkos::View<const int*> edge_receivers,
        Kokkos::View<const int*> edge_sources);
    void reduce_prepared_node_forces(std::uint64_t graph_generation);
    void reduce_stress(
        double volume,Kokkos::View<const double*> xyz);
    void reduce_prepared_stress(
        double volume,std::uint64_t graph_generation);
    void prepare_factorized_batch(
        std::uint64_t graph_generation,
        std::span<const std::int64_t> edge_offsets);
    void reduce_prepared_batched_stress(
        std::span<const double> volumes,
        std::uint64_t graph_generation);
    std::uint64_t factorized_graph_generation() const {
        return factorized_prepared_graph_generation;
    }
    std::size_t factorized_prepared_graph_count() const {
        return factorized_prepared_graphs;
    }
    std::size_t factorized_prepared_evaluation_count() const {
        return factorized_prepared_evaluations;
    }
    std::size_t factorized_topology_validation_count() const {
        return factorized_topology_validations;
    }
    std::size_t factorized_topology_validation_skip_count() const {
        return factorized_topology_validation_skips;
    }
    std::size_t factorized_preparation_fence_count() const {
        return factorized_preparation_fences;
    }
    std::size_t factorized_evaluation_fence_count() const {
        return factorized_evaluation_fences;
    }
    std::size_t factorized_stage_fence_count() const {
        return factorized_stage_fences;
    }
    std::size_t factorized_zbl_evaluator_stream_launch_count() const {
        return factorized_zbl_evaluator_stream_launches;
    }
    int execution_geometry_capacity_edges() const {
        return execution_geometry_edge_capacity;
    }
    const std::string& execution_geometry_growth_reason() const {
        return execution_geometry_growth_reason_;
    }
    void set_execution_geometry_device_memory_info_for_testing(
        std::size_t free_bytes,std::size_t total_bytes) {
        if(free_bytes>total_bytes)
            throw std::invalid_argument(
                "Execution geometry device free bytes exceed total bytes.");
        execution_geometry_memory_info_override_=total_bytes!=0;
        execution_geometry_device_free_=free_bytes;
        execution_geometry_device_total_=total_bytes;
    }
    std::size_t execution_geometry_workspace_bytes() const;
    std::size_t execution_geometry_allocation_count() const {
        return execution_geometry_allocations;
    }
    std::size_t execution_geometry_copy_count() const {
        return execution_geometry_copies;
    }
    std::size_t execution_sphericart_initialization_count() const {
        return execution_sphericart_initializations;
    }
    std::size_t execution_sphericart_launch_count() const {
        return execution_sphericart_launches;
    }
    std::size_t execution_sphericart_async_launch_count() const {
        return execution_sphericart_async_launches;
    }
    void set_execution_mh1_scratch_budget_bytes(std::size_t budget_bytes);
    std::size_t execution_mh1_scratch_budget_bytes() const {
        return execution_mh1_scratch_budget;
    }
    std::size_t execution_mh1_scratch_minimum_bytes() const {
        return execution_mh1_scratch_minimum;
    }
    std::size_t execution_mh1_scratch_planned_bytes() const {
        return execution_mh1_scratch_planned;
    }
    std::size_t execution_mh1_scratch_retained_bytes() const {
        return execution_mh1_scratch_retained;
    }
    std::size_t execution_mh1_scratch_recomputed_bytes() const {
        return execution_mh1_scratch_recomputed;
    }
    std::size_t execution_mh1_scratch_recomputed_layer_count() const {
        return execution_mh1_scratch_recomputed_layers;
    }
    bool execution_mh1_scratch_budget_satisfied() const {
        return execution_mh1_scratch_planned<=execution_mh1_scratch_budget;
    }
    std::size_t execution_mh1_conditioning_recomputation_count() const {
        return execution_mh1_conditioning_recomputations;
    }
    std::array<int,2> execution_mh1_retained_interaction_output_dimensions() const {
        return {
            execution_mh1_node_runtime[0].retained_interaction_output_dimension,
            execution_mh1_node_runtime[1].retained_interaction_output_dimension,
        };
    }
    std::size_t execution_mh1_generated_forward_launch_count() const {
        return execution_mh1_generated_forward_launches;
    }
    std::size_t execution_mh1_generated_source_reverse_launch_count() const {
        return execution_mh1_generated_source_reverse_launches;
    }
    std::size_t execution_mh1_generated_edge_reverse_launch_count() const {
        return execution_mh1_generated_edge_reverse_launches;
    }
    std::size_t execution_mh1_generated_conditioning_forward_launch_count() const {
        return execution_mh1_generated_conditioning_forward_launches;
    }
    std::size_t execution_mh1_generated_conditioning_reverse_launch_count() const {
        return execution_mh1_generated_conditioning_reverse_launches;
    }
    bool factorized_source_owned_reverse() const {
        return mace_uses_prepared_execution(streamed_edges)
            &&factorized_source_owned_reverse_used;
    }
    int edge_workspace_rows() const;
    std::size_t edge_workspace_bytes() const;
    std::size_t conditioned_mlp_workspace_bytes() const;
    void fence() const {
        factorized_execution_space.fence("MACE_Nonlinear public fence");
    }
    // Optional MPI hooks.  The engine never links MPI directly: the pair style
    // installs an in-place global-sum callback (MPI_Allreduce) and declares how
    // many leading graph nodes are owned by this rank.  Serial users leave both
    // unset, which preserves the original single-rank behavior.
    using GlobalSumCallback = std::function<void(double*, std::size_t)>;
    void set_global_sum_callback(GlobalSumCallback callback) {
        global_sum_callback_=std::move(callback);
    }
    void set_polar_owned_nodes(const int owned_nodes) {
        polar_owned_nodes_=owned_nodes;
    }
    // H1 multi-rank path: readout and polar seeds describe owned-node
    // energies only.  Ghost copies must not be seeded, otherwise exchanging
    // their adjoints back to the owner double counts those derivatives.
    void set_polar_mask_ghost_adjoint_seeds(const bool enabled) {
        polar_mask_ghost_adjoint_seeds_=enabled;
    }
    void set_polar_global_rank(const int rank) {
        polar_global_rank_=rank;
    }
    // Optional per-layer message-passing hooks.  The forward hook is called
    // before each interaction with the full node-feature array; the reverse
    // hook is called after each layer's adjoint is complete.  A multi-rank pair
    // style uses them to exchange ghost features/adjoints (H1-style MPI).
    using LayerSyncCallback = std::function<void(
        int, Kokkos::View<Precision**,Kokkos::LayoutRight>)>;
    void set_layer_forward_sync(LayerSyncCallback callback) {
        layer_forward_sync_=std::move(callback);
    }
    void set_layer_reverse_sync(LayerSyncCallback callback) {
        layer_reverse_sync_=std::move(callback);
    }
    int node_feature_width() const {
        // This value is the per-atom communication width declared to LAMMPS
        // before the first compute (the pair style copies it into
        // comm_forward/comm_reverse, and Comm::init sizes its send/recv
        // buffers from them once).  The H1 layer hooks exchange the *input*
        // feature array of every interaction layer: layer 0 exchanges the
        // node embedding, later layers the previous product block output.
        // Return the maximum, otherwise a wider later layer overruns the
        // buffers sized for the embedding width.
        int width=node_embedding.output_dimension();
        for(const auto& product:products)
            width=std::max(width,product.output_dimension());
        return width;
    }
    void set_e3_linear_backend(const std::string& backend);
    std::string e3_linear_backend() const;
    std::string selected_e3_linear_backend(std::size_t samples) const;
    void set_execution_mh1_host_node_backend(const std::string& backend);
    const std::string& execution_mh1_host_node_backend() const {
        return execution_mh1_host_node_backend_request;
    }
    std::string selected_execution_mh1_host_node_backend() const;
    std::string tensor_product_backend() const;
    std::string tensor_product_execution_backend() const;
    int tensor_product_channel_team_size() const;
    int tensor_product_harmonic_team_size() const;
    void set_fused_gate_normalization_reverse(bool enabled);
    bool fused_gate_normalization_reverse_available() const;
    bool uses_fused_gate_normalization_reverse() const;
    void set_direct_node_tensor_reverse(bool enabled);
    bool direct_node_tensor_reverse_available() const;
    bool uses_direct_node_tensor_reverse() const;
    std::size_t linear_workspace_bytes() const;
    std::size_t tensor_workspace_bytes() const;
    std::size_t product_workspace_bytes() const;
    std::size_t node_workspace_bytes() const;
    std::size_t precision_workspace_bytes() const;
    std::vector<int> atomic_numbers_host;
    GlobalSumCallback global_sum_callback_;
    LayerSyncCallback layer_forward_sync_;
    LayerSyncCallback layer_reverse_sync_;
    int polar_owned_nodes_=-1;
    bool polar_mask_ghost_adjoint_seeds_=false;
    int polar_global_rank_=0;
    int polar_owned_node_count(int num_nodes) const {
        return (polar_owned_nodes_>=0&&polar_owned_nodes_<=num_nodes)
            ?polar_owned_nodes_:num_nodes;
    }
    void polar_global_sum_host(double* values,std::size_t count) {
        if(global_sum_callback_)global_sum_callback_(values,count);
    }
    template<class View>
    void polar_global_sum_view(View view) {
        if(!global_sum_callback_)return;
        const std::size_t count=view.extent(0);
        if(count==0)return;
        std::vector<double> buffer(count);
        auto host=Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),view);
        for(std::size_t i=0;i<count;++i)
            buffer[i]=static_cast<double>(host(i));
        global_sum_callback_(buffer.data(),count);
        for(std::size_t i=0;i<count;++i)
            host(i)=static_cast<Precision>(buffer[i]);
        Kokkos::deep_copy(view,host);
    }
    Kokkos::View<int*> atomic_numbers;
    Kokkos::View<double*> atomic_energies,node_energies,node_forces,atom_forces,
        stress_tensor;
    // --- PolarMACE 中间量(P3 模块①;仅 polar_enabled 模型填充) ---
    // 行=节点;8 列平坦序 [sα, xα, yα, zα, sβ, xβ, yβ, zβ]
    // (= e3nn "1x0e+1x1o+1x0e+1x1o" 序列重复布局;依据 extensions.py L734 的 view(N,2,4))。
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_spin_charge_density,
        polar_spin_charge_density_storage;
    // P3 模块②: fukui_source_map 原始输出(N×2)与校正后初始密度(N×8,
    // 即 mace 的 field_independent_spin_charge_density,SCF/local_electron 的输入)。
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_fukui_sources,
        polar_fukui_sources_storage,polar_field_independent_density,
        polar_field_independent_density_storage;
    // P3 模块③④: k 网格 + GTO 几何预计算 + 势特征(几何缓存缓冲,每次求值重建)。
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_k_vectors;      // (Nk,3)
    Kokkos::View<Precision*> polar_k_norm2,polar_k0_mask;               // (Nk)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_cosines,
        polar_sines;                                                    // (Nk,N)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_density_basis_re,
        polar_density_basis_im;                                         // (Nk,4)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_feature_basis_re,
        polar_feature_basis_im;                                         // (Nk,8)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_potential_features;
                                                                        // (N,16) α|β
    // P3 模块⑤: SCF 密度更新(mixer → 更新块 → 逐步 Fukui 重校正)。
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_features_mixed,
        polar_features_mixed_storage;                                   // (N,2048)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_pe_out,polar_tp_out; // (N,2048)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_dp_out,polar_se_out,
        polar_nl_out;                                                   // (N,512)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_ro_l1_out,
        polar_ro_lmid_out;                                              // (N,160)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_gate1_out,
        polar_gate2_out;                                                // (N,128)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_update_out;     // (N,10)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_field_feats_a,
        polar_field_feats_b,polar_step_potential_features;              // (N,8)/(N,16)
    Kokkos::View<Precision***,Kokkos::LayoutRight> polar_scf_step_densities;
                                                                        // (S,N,8)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_scf_final_density;
                                                                        // (N,8)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_mixer_temp,
        polar_mixer_temp_storage,polar_scratch_a,polar_scratch_a_storage,
        polar_scratch_b,polar_scratch_b_storage,polar_nl_input,
        polar_nl_input_storage,polar_nl_a,polar_nl_a_storage,polar_nl_b,
        polar_nl_b_storage;                                             // ⑤暂存
    // ===== [v3 适配] 模块⑧ hpp 公开增补(反向入口/缓存/工作区; binding 与 pair style 共用) =====
    // --- P3 模块⑧: 力/全链反向 (2026-10-15; 前向缓存增量 + 反向工作区 + 节点力累加器) ---
    // 语义: node 级力累加器,写作 −∂E/∂r(n,·),与 node_forces(边级,−∂E/∂xyz_edge) 同符号约定;
    // 归约/消费点见 §5。仅标准 materialized 路径填充(polar 前向已拒绝 direct/prepared)。
    Kokkos::View<double*> polar_node_forces,polar_node_forces_storage;   // (3N,)
    // 种子(⑥/⑦ 反向段在 compute_polar_reverse_forces 之前投递;consumed-once,调用后清零):
    int polar_reverse_seed_terms=0;                                      // bit0=⑦le, bit1=⑥coulomb
    // ⑥ coulomb 反向开关:本提案默认 false(⑥ 未落码);⑥ 落码时在其 ctor 段置 true。
    bool polar_coulomb_enabled=false;
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_seed_density_adjoint,
        polar_seed_density_adjoint_storage;                              // (N,8) ∂E/∂d_final(末态密度)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_seed_density0_adjoint,
        polar_seed_density0_adjoint_storage;                             // (N,8) ∂E/∂d_init(初校正后密度)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_seed_potential_adjoint,
        polar_seed_potential_adjoint_storage;                            // (N,16) ∂E/∂P_last(末步势特征)
    // [dbg 2026-10-16] ⑧-7 le 反向 stage 探针(只读镜像;逐级对账用,不参与链)。
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_dbg_le_qin_bar,
        polar_dbg_le_qin_bar_storage;                                    // (N,8)  ∂E_le/∂q_in
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_dbg_le_plast_bar,
        polar_dbg_le_plast_bar_storage;                                  // (N,16) ∂E_le/∂P_last
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_dbg_le_nf_bar,
        polar_dbg_le_nf_bar_storage;                                     // (N,2048) ∂E_le/∂nf_last
    // [dbg 2026-10-16 v24] 反向链 stage 检查点(只读镜像;POLAR_DBG_CKPT 置位时写入)。
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_dbg_rev_dfinal,
        polar_dbg_rev_dfinal_storage;                                    // (N,8)  ∂E/∂d_final(循环前全种子)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_dbg_rev_out_last,
        polar_dbg_rev_out_last_storage;                                  // (N,10) ∂E/∂out(末步校正反向后)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_dbg_rev_out_first,
        polar_dbg_rev_out_first_storage;                                 // (N,10) ∂E/∂out(首步校正反向后)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_dbg_rev_dpost,
        polar_dbg_rev_dpost_storage;                                     // (N,8)  ∂E/∂d_post(末步迭代末)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_dbg_rev_dinit,
        polar_dbg_rev_dinit_storage;                                     // (N,8)  ∂E/∂d_init(sd0 并入后)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_dbg_rev_fmix,
        polar_dbg_rev_fmix_storage;                                      // (N,F)  ∂E/∂features_mixed
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_dbg_rev_pos_coul,
        polar_dbg_rev_pos_coul_storage;                                  // (N,3)  ⑥ 段后位置伴
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_dbg_rev_pos_all,
        polar_dbg_rev_pos_all_storage;                                   // (N,3)  全链位置伴(终值)
    // ---- P4 RBE (随机批 Ewald k 采样;依据 docs/design/RBE_P4_DESIGN.md) ----
    bool polar_rbe_enabled=false;                 // JSON "rbe".enabled(缺省关=确定性零改动)
    int polar_rbe_p=200;                          // 批大小 P
    double polar_rbe_sigma=1.5;                   // 采样包络 σ_q(=密度基宽)
    unsigned long long polar_rbe_seed=20261016ULL; // RNG 种子(入 JSON;ENV 可覆盖)
    long long polar_rbe_frame=0;                  // 帧计数(每 eval +1 ⇒ 帧间重采样)
    double polar_rbe_zq=1.0;                      // 离散归一 Σ_{m≠0} e^{−σ²k²/2}(精确直和)
    Kokkos::View<Precision*> polar_k_weight,
        polar_k_weight_storage;                   // (K,) 每 k 权重: 确定性=1;RBE=Z_q·e^{+σ²k²/2}/(2P)
    // 前向缓存增量(§2.1):
    // [v3 适配·D-02] 校后逐步密度直接复用既有 polar_scf_step_densities
    // (其快照内核位于逐步 Fukui 校正之后 = 校后口径; 提案的 _post 重复视图不新增)。
    std::array<Precision,6> polar_fukui_init_scalars{};                  // [Sα,Sβ,predα,predβ,Fα,Fβ]
    std::vector<std::array<Precision,6>> polar_scf_fukui_scalars;        // (S,6) 同上(逐步)
    double polar_cell_volume=1.0;                                        // 模块③ volume(宿主缓存)
    Precision polar_rcell_host[9]={};                                    // 模块③ rcell(宿主;维里预留)
    // cell 伴随占位(硬缺口 3;本 slice 仅清零与累加接口,消费者见 §6):
    Kokkos::View<Precision*> polar_cell_adjoint,polar_cell_adjoint_storage;  // (9,)
    // 反向工作区(首次调用 ensure_view 按对象维度分配;命名沿 §4):
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_rev_density_adjoint,
        polar_rev_density_adjoint_storage,                               // (N,8) d̄ 链
        polar_rev_channel_adjoint,polar_rev_channel_adjoint_storage,     // (N,8) 逐步 d̄ 汇入
        polar_rev_source_adjoint,polar_rev_source_adjoint_storage,       // (N,8) ⑥ q̄(用 0..3 列)
        polar_rev_feats_adjoint,polar_rev_feats_adjoint_storage,         // (N,8) 单通道 P̄ 拆分
        polar_rev_potential_adjoint,polar_rev_potential_adjoint_storage, // (N,16) P̄
        polar_rev_position_adjoint,polar_rev_position_adjoint_storage,   // (N,3) r̄
        polar_rev_nfeat_adjoint,polar_rev_nfeat_adjoint_storage,         // (N,2048) e3nn node-feat 伴
        polar_rev_fmix_adjoint,polar_rev_fmix_adjoint_storage,           // (N,2048)
        polar_rev_uvu_mixed_bar,polar_rev_uvu_mixed_bar_storage,         // (N,mul1) mode1
        polar_rev_le_mlp_out_bar,polar_rev_le_mlp_out_bar_storage,       // (N,1)
        polar_rev_le_invar_bar,polar_rev_le_invar_bar_storage,           // (N,1024)
        polar_rev_le_dot_q_bar,polar_rev_le_dot_q_bar_storage,           // (N,512)
        polar_rev_le_dot_v_bar,polar_rev_le_dot_v_bar_storage,           // (N,512)
        polar_rev_le_q_up_bar,polar_rev_le_q_up_bar_storage,             // (N,512)
        polar_rev_le_v_up_bar,polar_rev_le_v_up_bar_storage,             // (N,512)
        polar_rev_le_nf_bar,polar_rev_le_nf_bar_storage,                 // (N,2048)
        polar_rev_le_q_in_bar,polar_rev_le_q_in_bar_storage,             // (N,8)
        polar_rev_le_p_last_bar,polar_rev_le_p_last_bar_storage,         // (N,16)
        polar_rev_fukui_source_adjoint,polar_rev_fukui_source_adjoint_storage, // (N,2)
        polar_rev_out_adjoint,polar_rev_out_adjoint_storage,             // (N,10)
        polar_rev_gate1_adjoint,polar_rev_gate1_adjoint_storage,         // (N,gate_w)
        polar_rev_gate2_adjoint,polar_rev_gate2_adjoint_storage,         // (N,gate_w)
        polar_rev_ro1_adjoint,polar_rev_ro1_adjoint_storage,             // (N,ro_w)
        polar_rev_ro2_adjoint,polar_rev_ro2_adjoint_storage,             // (N,ro_w)
        polar_rev_tp_adjoint,polar_rev_tp_adjoint_storage,               // (N,tp_w)
        polar_rev_nl_adjoint,polar_rev_nl_adjoint_storage,               // (N,hidden) 滚动激活伴
        polar_rev_h_adjoint,polar_rev_h_adjoint_storage,                 // (N,hidden) LN 输入伴
        polar_rev_nlin_adjoint,polar_rev_nlin_adjoint_storage,           // (N,nl_in_w)
        polar_rev_dp_adjoint,polar_rev_dp_adjoint_storage,               // (N,512)
        polar_rev_se_adjoint,polar_rev_se_adjoint_storage,               // (N,512)
        polar_rev_pe_adjoint,polar_rev_pe_adjoint_storage;               // (N,pe_w)
    // round-3 UVU reverse split-reduction partials: (S,N,mul*d1)
    Kokkos::View<Precision***,Kokkos::LayoutRight> polar_rev_uvu_partial_x1,
        polar_rev_uvu_partial_x1_storage,
        polar_rev_uvu_partial_x2,polar_rev_uvu_partial_x2_storage;
    // round-3 forward/replay/LE UVU split-reduction partials: (S,N,mul1)
    Kokkos::View<Precision***,Kokkos::LayoutRight> polar_fwd_uvu_partial,
        polar_fwd_uvu_partial_storage;
    // 重放前向值(不触碰⑤导出视图):
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_rev_step_potential,
        polar_rev_step_potential_storage,                                // (N,16)
        polar_rev_pe,polar_rev_pe_storage,                               // (N,pe_w)
        polar_rev_dp,polar_rev_dp_storage,                               // (N,512)
        polar_rev_se,polar_rev_se_storage,                               // (N,512)
        polar_rev_nl_in,polar_rev_nl_in_storage,                         // (N,nl_in_w)
        polar_rev_nl_h,polar_rev_nl_h_storage,                           // (N,3*hidden)
        polar_rev_nl_a,polar_rev_nl_a_storage,                           // (N,3*hidden)
        polar_rev_nl_out,polar_rev_nl_out_storage,                       // (N,nl_out_w)
        polar_rev_tp,polar_rev_tp_storage,                               // (N,tp_w)
        polar_rev_ro1,polar_rev_ro1_storage,                             // (N,ro_w)
        polar_rev_ro2,polar_rev_ro2_storage,                             // (N,ro_w)
        polar_rev_gate1,polar_rev_gate1_storage,                         // (N,gate_w)
        polar_rev_gate2,polar_rev_gate2_storage,                         // (N,gate_w)
        polar_rev_out,polar_rev_out_storage,                             // (N,10)
        polar_rev_nlout_adjoint,polar_rev_nlout_adjoint_storage;         // (N,nl_out_w)
    // Plan B1 (2026-10-20): per-step forward state cached for the SCF reverse
    // so the reverse does not replay ③④⑤.  Populated in run_polar_pipeline
    // only when MATOS_REPLAY_CACHE=1; compute_polar_reverse_forces then loads
    // these into polar_rev_* instead of calling replay_step_forward.  Layout
    // (steps,num_nodes,width); ~6.7 MB at 128 nodes x 2 steps (FP64).
    Kokkos::View<Precision***,Kokkos::LayoutRight> polar_rev_cache_pe,
        polar_rev_cache_pe_storage,                                      // (S,N,pe_w)
        polar_rev_cache_nl_h,polar_rev_cache_nl_h_storage,               // (S,N,3h)
        polar_rev_cache_nl_a,polar_rev_cache_nl_a_storage,               // (S,N,3h)
        polar_rev_cache_nl_out,polar_rev_cache_nl_out_storage,           // (S,N,512)
        polar_rev_cache_ro1,polar_rev_cache_ro1_storage,                 // (S,N,160)
        polar_rev_cache_ro2,polar_rev_cache_ro2_storage,                 // (S,N,160)
        polar_rev_cache_out,polar_rev_cache_out_storage;                 // (S,N,10)
    bool polar_rev_cache_valid=false;   // false ⇒ reverse must use replay
    // 诊断:最近一次反向实际消费的种子掩码(0=无能量项⇒零力;3=⑥+⑦)。非 consume-once。
    int polar_reverse_terms_last=0;
    // k 空间反向暂存:
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_rev_ab,
        polar_rev_ab_storage;                                            // (Nk,16)=[Ā|Ŷ] 投影反向
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_rev_ab_partial,
        polar_rev_ab_partial_storage;                                    // (Nk,8*S*2) round-3 split
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_rev_phase_partial,
        polar_rev_phase_partial_storage;                                 // (N,S*3) round-3 split
    Kokkos::View<Precision**,Kokkos::LayoutRight>
        polar_rev_phase_source_partial,
        polar_rev_phase_source_partial_storage;                          // (N,S*7) round-3 split
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_rev_acc_cos,
        polar_rev_acc_cos_storage,polar_rev_acc_sin,polar_rev_acc_sin_storage;
                                                                         // (Nk,4) assemble 反向
    Kokkos::View<Precision*> polar_rev_rho_adj_re,polar_rev_rho_adj_re_storage,
        polar_rev_rho_adj_im,polar_rev_rho_adj_im_storage,                // (Nk)
        polar_rev_rho_fwd_re,polar_rev_rho_fwd_re_storage,
        polar_rev_rho_fwd_im,polar_rev_rho_fwd_im_storage,                // (Nk) ⑥ 重算 ρ
        polar_rev_transform,polar_rev_transform_storage;                  // (8,) Fukui 反向系数
    // 反向入口:必须在 readout 播种之后、骨架反向主循环之前调用(§4.0-A)。
    void compute_polar_reverse_forces(int num_nodes);
    // ⑧-A 公开投递口(⑦ le 反向的 n̄f_last 路径;内部 init=false 累加;定义见 §4.2 尾)。
    void polar_inject_node_features_adjoint(int layer,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> adjoint);
    // pair style 消费口(§5.2;binding §7.1 同源)。
    bool polar_blocks_enabled() const { return polar_enabled; }
    // ===== [v2 适配] 模块⑥⑦ hpp 公开增补(binding 与 run_polar_pipeline 共用) =====
    // [公开(供 binding)] ⑦ 运行期视图(module7_patch_final.md §4.1a 原文; 注释 (N,512) 按预检 §8-1 保留):
    // P3 模块⑦: local_electron_energy（OneBodyMLPFieldReadout; 仅 polar_le_enabled 填充）
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_le_q_in,
        polar_le_q_in_storage;                                          // (N,8)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_le_q_up,
        polar_le_q_up_storage,polar_le_v_up,polar_le_v_up_storage;      // (N,512)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_le_dot_q,
        polar_le_dot_q_storage,polar_le_dot_v,polar_le_dot_v_storage;   // (N,512)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_le_invar,
        polar_le_invar_storage;                                         // (N,1024)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_le_mlp_out,
        polar_le_mlp_out_storage;                                       // (N,1)
    Kokkos::View<Precision*> polar_le_atom,polar_le_atom_storage;       // (N,)
    Precision polar_le_total=Precision(0);                              // 末次求值总量
    // [公开(供 binding)] ⑥ 视图与配置(module6_patch_final.md §1(补丁 A)原文; si 表/bool 随提案同区):
    // --- P3 模块⑥: coulomb_energy(全 k GTO-Ewald;前向标量 eV + ρ̂(k) 诊断) ---
    // ρ̂ = 未乘 (2π)³/V 的傅里叶密度(即 features.py L58-63 的括号项);
    // E = FIELD_CONSTANT/(2V)·Σ_k 2·(ρ̂_r²+ρ̂_i²)/|k|²(k=0 跳过)。
    bool polar_include_self_interaction=true,polar_include_pbc_corrections=true;
    Kokkos::View<Precision*> polar_coulomb_si_overlap;                   // (4)
    Kokkos::View<int*> polar_coulomb_si_select;                          // (4)
    Kokkos::View<Precision*> polar_coulomb_rho_re,polar_coulomb_rho_im,
        polar_coulomb_rho_re_storage,polar_coulomb_rho_im_storage;       // (Nk)
    Kokkos::View<double*> polar_coulomb_energy,polar_coulomb_energy_storage;
                                                                        // (1) eV
    void compute_node_energies_forces(int num_nodes,Kokkos::View<const int*> node_types,
        Kokkos::View<const int*> num_neigh,Kokkos::View<const int*> neigh_indices,
        Kokkos::View<const int*> neigh_types,Kokkos::View<const double*> xyz,
        Kokkos::View<const double*> distances,
        std::uint64_t execution_graph_generation=0,
        double total_charge=0.0,double total_spin=1.0,
        Kokkos::View<const double*> node_positions={},
        Kokkos::View<const double*> cell={});
    void compute_Y(
        Kokkos::View<const double*> xyz,bool evaluator_stream=false);
    // P3 极性静电场全链(定义在独立 TU mace_polar_impl.tpp; ②③④⑤及后续⑥⑦⑧)。
    void run_polar_pipeline(int num_nodes,
        Kokkos::View<Precision**,Kokkos::LayoutRight> node_attrs,
        Kokkos::View<const double*> node_positions,
        Kokkos::View<const double*> cell,
        double total_charge,double total_spin);
    // [v2 适配] 模块⑥⑦ 扩展表解析(定义于独立 TU mace_polar_impl.tpp; ctor 于 polar 解析块收尾调用)。
    void parse_polar_extension_tables();

private:
    struct SphericalHarmonicsState;
    explicit MaceNonlinearKokkosT(const nlohmann::json& data);

public:
    struct Gate {
        int input_size=0,scalar_size=0,gate_size=0,gated_size=0,output_size=0;
        std::vector<IrrepBlock> scalar_blocks,gated_blocks;
        std::vector<Precision> scalar_constants;
        std::vector<Precision> gate_constants;
        Kokkos::View<Precision*> fused_scalar_constants,fused_gate_constants;
        Kokkos::View<int**,Kokkos::LayoutRight> fused_gate_plan,
            fused_packed_input_plan;
        explicit Gate(const nlohmann::json& data);
        bool supports_fused_normalized_reverse() const;
        void evaluate(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
                      Kokkos::View<Precision**,Kokkos::LayoutRight> output) const;
        void evaluate_normalized_to_packed(
            Kokkos::View<const Precision**,Kokkos::LayoutRight> linear_forward,
            Kokkos::View<const Precision**,Kokkos::LayoutRight> residual,
            Kokkos::View<const int*> residual_active,
            Kokkos::View<const Precision*> densities,Precision alpha,Precision beta,
            Kokkos::View<Precision**,Kokkos::LayoutRight> input,
            Kokkos::View<Precision**,Kokkos::LayoutRight> packed_output,
            bool linear_is_normalized=false,
            Kokkos::View<const Precision*> normalization_inverse={}) const;
        void reverse(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
                     Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
                     Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint) const;
        void reverse_normalized(
            Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
            Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
            Kokkos::View<const Precision**,Kokkos::LayoutRight> linear_forward,
            Kokkos::View<const Precision*> densities,Precision alpha,Precision beta,
            Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint,
            Kokkos::View<Precision**,Kokkos::LayoutRight> linear_adjoint,
            Kokkos::View<Precision*> density_adjoint) const;
        void reverse_normalized_packed(
            Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
            Kokkos::View<const Precision**,Kokkos::LayoutRight> packed_output_adjoint,
            Kokkos::View<const Precision**,Kokkos::LayoutRight> packed_linear_forward,
            Kokkos::View<const Precision*> densities,Precision alpha,Precision beta,
            Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint,
            Kokkos::View<Precision**,Kokkos::LayoutRight> packed_linear_adjoint,
            Kokkos::View<Precision*> density_adjoint,
            bool linear_is_normalized=false) const;
    };
private:
    struct Interaction {
        E3LinearKokkosT<Precision> source_embedding,target_embedding,linear_up,skip,linear_res,linear_1,linear_2;
        E3TensorProductKokkosT<Precision> convolution;
        AffineMLPKokkosT<Precision> convolution_weights,density;
        Gate gate;
        Kokkos::View<Precision**,Kokkos::LayoutRight> convolution_source_contributions;
        Kokkos::View<Precision**,Kokkos::LayoutRight> convolution_target_contributions;
        Kokkos::View<Precision**,Kokkos::LayoutRight> density_source_contributions;
        Kokkos::View<Precision**,Kokkos::LayoutRight> density_target_contributions;
        RadialFunctionSetKokkos<Precision> pair_spline;
        int pair_spline_weight_count=0;
        bool pair_spline_ready=false;
        Precision alpha,beta;
        explicit Interaction(const nlohmann::json& data);
        bool prepare_pair_conditioning(
            const nlohmann::json& data,int radial_size,int model_element_count,
            const std::vector<int>& selected_model_indices);
        bool prepare_pair_spline(
            const nlohmann::json& data,const nlohmann::json& radial_data,
            double cutoff,int radial_size,int model_element_count,
            const std::vector<int>& selected_model_indices,
            const std::vector<int>& model_atomic_numbers,int spline_nodes);
        void set_e3_linear_backend(const std::string& backend);
        void set_e3_linear_workspace(
            const std::shared_ptr<typename E3LinearKokkosT<Precision>::Workspace>& workspace);
        void set_conditioned_mlp_workspace(
            const std::shared_ptr<typename AffineMLPKokkosT<Precision>::Workspace>&
                workspace);
        std::size_t linear_workspace_bytes() const;
    };
    void prepare_mh1_pair_spline_for_graph(
        Kokkos::View<const int*> node_types,
        Kokkos::View<const int*> neigh_types);
public:
    struct Readout {
        bool nonlinear=false,has_mid=false; E3LinearKokkosT<Precision> linear,linear_1,linear_2,linear_mid; Precision activation_constant=1.0;
        Kokkos::View<Precision**,Kokkos::LayoutRight> hidden,activated,result,seed,
            activated_adj,hidden_adj;
        Kokkos::View<Precision**,Kokkos::LayoutRight> hidden_mid,activated_mid,mid_adj,act_mid_adj;
        Kokkos::View<Precision**,Kokkos::LayoutRight> hidden_storage,activated_storage,
            result_storage,seed_storage,activated_adj_storage,hidden_adj_storage;
        Kokkos::View<Precision**,Kokkos::LayoutRight> hidden_mid_storage,activated_mid_storage,
            mid_adj_storage,act_mid_adj_storage;
        explicit Readout(const nlohmann::json& data);
        void set_e3_linear_backend(const std::string& backend);
        void set_e3_linear_workspace(
            const std::shared_ptr<typename E3LinearKokkosT<Precision>::Workspace>& workspace);
        std::size_t linear_workspace_bytes() const;
        void evaluate(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,Kokkos::View<Precision*> output);
        // 与 evaluate 同链,但复制 final linear 的全部列(供 2x0e 等多通道读出头,
        // 如 PolarMACE 的 fukui_source_map;evaluate 只取第 0 列)。
        void evaluate_all(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
                          Kokkos::View<Precision**,Kokkos::LayoutRight> output);
        // 模块⑧(R1): evaluate_all 的伴随。seed_all(N×k) = ∂E/∂output(k=linear_2 输出列数,
        // fukui 头 k=2);中间伴覆盖写,最终伴按 initialize_input_adjoint 选择清零/累加
        // (进 states[ℓ].layer_adjoint 必须 false —— 播种为覆盖写,见 mace_nonlinear_kokkos.hpp
        //  polar 段注释)。依赖:调用前必须先对同一实例执行过 evaluate_all(缓存 hidden/
        // hidden_mid),与现有 reverse 的隐藏依赖完全同构。
        void reverse_all(Kokkos::View<const Precision**,Kokkos::LayoutRight> seed_all,
                         Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint,
                         bool initialize_input_adjoint);
        void reverse(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,Precision scale,
                     Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint);
    };
private:
    struct LayerState {
        Kokkos::View<Precision**,Kokkos::LayoutRight> input,up,residual,skip,messages,linear_1_output,pre_gate,gated,interaction_output,output;
        Kokkos::View<Precision**,Kokkos::LayoutRight> source_embeddings,target_embeddings,edge_features,raw_weights,weights,edge_up,edge_messages;
        Kokkos::View<Precision**,Kokkos::LayoutRight> execution_embedding,
            execution_embedding_adj,execution_graph_embedding,
            execution_graph_embedding_adj,execution_graph_fixed_contribution,
            execution_graph_harmonic_adj,execution_input_ir_mul,
            execution_output_ir_mul,execution_input_adjoint_ir_mul;
        Kokkos::View<Precision**,Kokkos::LayoutRight> convolution_contributions,density_contributions;
        Kokkos::View<Precision**,Kokkos::LayoutRight> density_matrix,layer_adjoint,
            interaction_output_adj,skip_adj,gated_adj,pre_gate_adj,linear_adj,
            message_adj,up_adj,edge_message_adj,edge_up_adj,
            edge_harmonic_adj,weight_adj,raw_weight_adj,edge_feature_adj,
            density_raw_adj,density_feature_adj,
            cutoff_partial,cutoff_partial_storage;
        Kokkos::View<Precision*> density_raw,density_base,densities,density_adj,
            normalization_inverse,execution_cutoff_adj,
            execution_graph_cutoff_adj;
        Kokkos::View<int*> agnostic_elements;

        Kokkos::View<Precision**,Kokkos::LayoutRight> up_storage,residual_storage,
            skip_storage,messages_storage,linear_1_output_storage,pre_gate_storage,
            gated_storage,interaction_output_storage,output_storage,
            source_embeddings_storage,target_embeddings_storage,
            edge_features_storage,raw_weights_storage,weights_storage,
            edge_up_storage,edge_messages_storage,convolution_contributions_storage,
            execution_embedding_storage,execution_embedding_adj_storage,
            execution_graph_embedding_storage,execution_graph_embedding_adj_storage,
            execution_graph_fixed_contribution_storage,
            execution_graph_harmonic_adj_storage,execution_input_ir_mul_storage,
            execution_output_ir_mul_storage,execution_input_adjoint_ir_mul_storage,
            density_contributions_storage,density_matrix_storage,
            layer_adjoint_storage,interaction_output_adj_storage,skip_adj_storage,
            gated_adj_storage,pre_gate_adj_storage,linear_adj_storage,
            message_adj_storage,up_adj_storage,
            edge_message_adj_storage,edge_up_adj_storage,edge_harmonic_adj_storage,
            weight_adj_storage,raw_weight_adj_storage,edge_feature_adj_storage,
            density_raw_adj_storage,density_feature_adj_storage;
        Kokkos::View<Precision*> density_raw_storage,density_base_storage,
            densities_storage,density_adj_storage,normalization_inverse_storage,
            execution_cutoff_adj_storage,
            execution_graph_cutoff_adj_storage;
        Kokkos::View<int*> agnostic_elements_storage;
    };
    struct ExecutionMH1NodeRuntime {
        static constexpr int capacity_tile_rows=256;
        static constexpr int cuda_throughput_tile_rows=1024;
        Kokkos::View<Precision*> linear_parameters,product_parameters,
            readout_parameters;
        Kokkos::View<Precision**,Kokkos::LayoutRight> arena,residual,
            linear_1_output,pre_gate,gated,interaction_output,
            interaction_adjoint,gated_adjoint,pre_gate_adjoint,
            linear_1_adjoint;
        int arena_dimension=0;
        int tile_rows=capacity_tile_rows;
        int retained_pre_gate_dimension=0;
        int retained_interaction_output_dimension=0;
        bool reuse_message_adjoint=false;
        std::size_t bytes() const {
            return sizeof(Precision)*(linear_parameters.size()
                +product_parameters.size()+readout_parameters.size()
                +arena.size()+residual.size()+linear_1_output.size()
                +pre_gate.size()+gated.size()+interaction_output.size()
                +interaction_adjoint.size()+gated_adjoint.size()
                +pre_gate_adjoint.size()+linear_1_adjoint.size());
        }
    };
    int l_max=0,num_lm=0,model_num_elements=0,cutoff_power=0,num_bessel=0;
    bool apply_cutoff=false,has_agnesi=false,has_zbl=false,mh1_fast_path=false;
    bool mh1_compiled_products=false,mh1_pair_conditioning=false,
        mh1_external_uvu_tensors=false;
    bool fused_gate_normalization_reverse_enabled=true;
    bool mh1_pair_spline_ready_=false;
    bool mh1_pair_spline_lazy_=false;
    // Format-v3 artifacts before this metadata existed retain the 256-node
    // default; extracted artifacts can select a different resolution.
    int mh1_pair_spline_nodes_=256;
    std::vector<int> mh1_selected_model_indices_host_;
    std::vector<int> mh1_model_atomic_numbers_host_;
    nlohmann::json mh1_pair_spline_interactions_data_;
    nlohmann::json mh1_pair_spline_radial_data_;
    std::vector<int> mh1_spline_global_types_;
    std::vector<int> mh1_spline_global_to_active_;
    Kokkos::View<int*> mh1_spline_node_types_;
    Kokkos::View<int*> mh1_spline_neigh_types_;
    // Persistent backing storage for the compacted per-eval type views, plus
    // the device-side global->active map.  Rebuilt on device each evaluation
    // so only a small presence mask crosses the host boundary.
    Kokkos::View<int*> mh1_spline_node_types_storage_;
    Kokkos::View<int*> mh1_spline_neigh_types_storage_;
    Kokkos::View<int*> mh1_spline_active_map_;
    Kokkos::View<int*> mh1_spline_type_mask_;
    int mh1_spline_type_count_=0;
    std::string mh1_edge_executor_="mlp_reference";
    std::string execution_mh1_node_arena_policy_="throughput-v1";
    int execution_mh1_node_arena_tile_rows_override_=0;
    int selected_execution_mh1_node_arena_tile_rows() const;
    void apply_execution_mh1_node_arena_tile_rows();
    bool uses_packed_node_linears() const;
    bool direct_node_tensor_reverse_enabled=true;
    std::string execution_mh1_host_node_backend_request="auto";
    MaceMH1FamilyDescriptor mh1_family;
    std::string mh1_fast_path_rejection;
    MACEStreamedEdgesMode streamed_edges=MACEStreamedEdgesMode::materialized;
    Kokkos::DefaultExecutionSpace factorized_execution_space;
    std::size_t factorized_forward_evaluations=0;
    std::size_t factorized_reverse_evaluations=0;
    std::size_t factorized_fallback_evaluations=0;
    std::size_t factorized_schedule_builds=0;
    std::uint64_t factorized_graph_generation_counter=0;
    std::uint64_t factorized_prepared_graph_generation=0;
    std::uint64_t factorized_prepared_geometry_graph_generation=0;
    std::uint64_t factorized_completed_evaluation_graph_generation=0;
    std::size_t factorized_prepared_graphs=0;
    std::size_t factorized_prepared_evaluations=0;
    std::size_t factorized_topology_validations=0;
    std::size_t factorized_topology_validation_skips=0;
    std::size_t factorized_preparation_fences=0;
    std::size_t factorized_evaluation_fences=0;
    std::size_t factorized_stage_fences=0;
    std::size_t factorized_zbl_evaluator_stream_launches=0;
    int execution_geometry_edge_capacity=0;
    std::string execution_geometry_growth_reason_="geometric growth";
    bool execution_geometry_memory_info_override_=false;
    std::size_t execution_geometry_device_free_=0;
    std::size_t execution_geometry_device_total_=0;
    std::size_t execution_geometry_allocations=0;
    std::size_t execution_geometry_copies=0;
    std::size_t execution_sphericart_initializations=0;
    std::size_t execution_sphericart_launches=0;
    std::size_t execution_sphericart_async_launches=0;
    std::size_t execution_mh1_scratch_budget=
        std::numeric_limits<std::size_t>::max();
    std::size_t execution_mh1_scratch_minimum=0;
    std::size_t execution_mh1_scratch_planned=0;
    std::size_t execution_mh1_scratch_retained=0;
    std::size_t execution_mh1_scratch_recomputed=0;
    std::size_t execution_mh1_scratch_recomputed_layers=0;
    std::size_t execution_mh1_conditioning_recomputations=0;
    std::vector<unsigned char> execution_mh1_retain_graph_embedding;
    std::vector<int> execution_mh1_graph_embedding_scratch_slot;
    std::vector<int> execution_mh1_graph_embedding_scratch_widths;
    std::vector<Kokkos::View<Precision**,Kokkos::LayoutRight>>
        execution_mh1_graph_embedding_scratch_storage,
        execution_mh1_graph_embedding_adjoint_scratch_storage;
    Kokkos::View<Precision**,Kokkos::LayoutRight>
        execution_mh1_graph_harmonic_adjoint_scratch_storage;
    Kokkos::View<Precision*> execution_mh1_graph_cutoff_adjoint_scratch_storage,
        execution_mh1_message_adjoint_scratch_storage,
        execution_mh1_up_adjoint_scratch_storage,
        execution_mh1_input_adjoint_scratch_storage,
        execution_mh1_density_adjoint_scratch_storage;
    bool execution_source_schedule_dirty=true;
    std::size_t execution_mh1_generated_forward_launches=0;
    std::size_t execution_mh1_generated_source_reverse_launches=0;
    std::size_t execution_mh1_generated_edge_reverse_launches=0;
    std::size_t execution_mh1_generated_conditioning_forward_launches=0;
    std::size_t execution_mh1_generated_conditioning_reverse_launches=0;
    bool factorized_source_owned_reverse_used=false;
    bool execution_mh1_has_model_contract=false;
    bool execution_mh1_has_model_contract_v4=false;
    std::string execution_mh1_model_generation_fingerprint;
    std::string execution_mh1_model_semantic_fingerprint;
    std::string execution_mh1_model_structure_fingerprint;
    std::string execution_mh1_model_generation_fingerprint_v4;
    std::string execution_mh1_model_semantic_fingerprint_v4;
    std::string execution_mh1_model_structure_fingerprint_v4;
    std::string execution_mh1_model_runtime_layout_fingerprint_v4;
    std::unique_ptr<symmetrix::execution::MH1HostPlugin> jit_mh1_host_plugin;
    std::unique_ptr<symmetrix::execution::MH1HostPluginV4>
        jit_mh1_host_plugin_v4;
    std::unique_ptr<symmetrix::execution::MH1HostPluginV5>
        jit_mh1_host_plugin_v5;
    std::unique_ptr<symmetrix::execution::MH1CudaPlugin> jit_mh1_cuda_plugin;
    std::unique_ptr<symmetrix::execution::MH1CudaPluginV4>
        jit_mh1_cuda_plugin_v4;
#ifdef KOKKOS_ENABLE_CUDA
    static constexpr int streamed_edge_block_size=16384;
#else
    static constexpr int streamed_edge_block_size=1024;
#endif
    bool streams_layer(int layer) const;
    void release_layer_edge_workspace(int layer);
public:
    // nvcc requires functions enclosing Kokkos extended lambdas to be public.
    void prepare_execution_source_schedule(
        int num_nodes,int samples,Kokkos::View<const int*> source_indices,
        Kokkos::View<const int*> target_indices);
private:
    void invalidate_factorized_prepared_graph();
    void reserve_execution_geometry_workspace(int edges);
    bool prepare_execution_source_schedule_host(
        int num_nodes,std::span<const int> source_indices,
        std::span<const int> target_indices);
    double radial_prefactor=0.0,agnesi_a=0.0,agnesi_q=0.0,agnesi_p=0.0,scale=1.0,shift=0.0;
    Kokkos::View<int*> model_indices,model_atomic_numbers;
    Kokkos::View<Precision*> bessel_weights,covalent_radii,cutoffs,Y,Y_grad,xyz_shuffled,Y_grad_shuffled;
    Kokkos::View<Precision**,Kokkos::LayoutRight> attrs,radial,edge_harmonics,features;
    Kokkos::View<int*> targets,product_elements,offsets;
    Kokkos::View<int*> execution_prepared_node_types,
        execution_prepared_num_neigh,execution_prepared_neigh_indices,
        execution_prepared_neigh_types,execution_prepared_offsets,
        execution_prepared_targets;
    Kokkos::View<double*> execution_prepared_xyz,execution_prepared_distances,
        execution_prepared_reference_positions,execution_prepared_reference_xyz,
        execution_prepared_cell,execution_prepared_inverse_cell,
        execution_prepared_positions,execution_prepared_displacements;
    Kokkos::View<int*> execution_prepared_pbc,
        execution_prepared_geometry_invalid;
    Kokkos::View<std::int64_t*> execution_prepared_batch_edge_offsets;
    Kokkos::View<double*> execution_prepared_batch_volumes;
    Kokkos::View<int*> execution_block_source_segments,
        execution_source_edge_offsets,execution_source_edges,
        execution_block_receivers,execution_graph_source_edge_offsets,
        execution_graph_source_edges,execution_graph_receivers,
        execution_mh1_spline_intervals_storage;
    Kokkos::View<Precision*> cutoffs_storage,Y_storage,Y_grad_storage,
        xyz_shuffled_storage,Y_grad_shuffled_storage,readout_contribution,
        readout_contribution_storage,cutoff_adjoints,cutoff_adjoints_storage,
        pair_spline_distance_adjoints,pair_spline_distance_adjoints_storage,
        execution_mh1_spline_coordinates_storage;
    Kokkos::View<double*> node_energies_storage,node_forces_storage,
        atom_forces_storage,stress_tensor_storage,
        zbl_energies,zbl_energies_storage,zbl_forces,zbl_forces_storage;
    Kokkos::View<Precision**,Kokkos::LayoutRight> attrs_storage,radial_storage,
        edge_harmonics_storage,features_storage,radial_adjoints,
        radial_adjoints_storage,harmonic_adjoints,harmonic_adjoints_storage;
    Kokkos::View<int*> targets_storage,product_elements_storage,offsets_storage;
    Kokkos::View<int*> execution_block_source_segments_storage,
        execution_source_edge_offsets_storage,execution_source_edges_storage,
        execution_block_receivers_storage,execution_graph_source_edge_offsets_storage,
        execution_graph_source_edges_storage,execution_graph_receivers_storage;
    int execution_graph_source_owner_count=0;
    int execution_graph_receiver_count=0;
    std::vector<int> execution_block_source_segments_host;
    std::vector<int> execution_block_receiver_counts_host;
    std::vector<int> execution_schedule_sources_host;
    std::vector<int> execution_schedule_targets_host;
    int execution_schedule_num_nodes=0;
    std::vector<int> execution_prepared_node_types_host;
    std::vector<std::int64_t> execution_prepared_batch_edge_offsets_host;
    std::uint64_t factorized_prepared_batch_graph_generation=0;
    std::vector<int> execution_prepared_num_neigh_host;
    std::vector<int> execution_prepared_neigh_indices_host;
    std::vector<int> execution_prepared_neigh_types_host;
    std::shared_ptr<typename E3LinearKokkosT<Precision>::Workspace>
        e3_linear_workspace;
    std::shared_ptr<typename AffineMLPKokkosT<Precision>::Workspace>
        conditioned_mlp_workspace;
    E3LinearKokkosT<Precision> node_embedding;
    std::vector<Interaction> interactions;
    std::vector<E3ProductBasisKokkosT<Precision>> products;
    std::vector<Readout> readouts;
    // --- PolarMACE 静电层 ("polar_blocks" 段;可选) ---
    // 首段仅解析 lr_source_maps + fukui_source_map (P3 模块①②);其余子段暂存
    // polar_json,按执行规格 (§12) 逐模块实现对账后再逐段接入。
    bool polar_enabled=false;
    nlohmann::json polar_json;
    std::vector<E3LinearKokkosT<Precision>> polar_lr_source_maps;
    std::optional<Readout> polar_fukui_source_map;  // Readout 仅 json 构造
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_lr_contribution,
        polar_lr_contribution_storage;  // 模块① 逐层求值暂存
    // 引擎内部 node-feats 的 l=1 分量基与 mace/e3nn 基相差固定矩阵 -P(P=(2,0,1)
    // 循环;含整体 -1;多构型数值实证残差 ~1e-8)。供 polar 块的 node_feats 须先
    // 经该列映射转换:e3nn 列 c 的取值为 -(内部列 (c+1)%3)。映射在 ctor 内按
    // lr_source_maps[0].irreps_in 构造(仅 l=0 直通与 l=1 循环,l>1 拒绝)。
    int polar_feature_width=0;
    Kokkos::View<int*> polar_basis_source;
    Kokkos::View<Precision*> polar_basis_sign;
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_node_feats_e3nn,
        polar_node_feats_e3nn_storage;
    // --- P3 模块③④ 表与配置(ctor 解析;公式经 numpy 预检对参考逐位锁定) ---
    Precision polar_kspace_cutoff=0.0;
    int polar_num_feature_sigmas=0;
    Kokkos::View<Precision*> polar_tab_density_cl,polar_tab_density_sigma2,
        polar_tab_density_pref0,polar_tab_density_pref1,
        polar_tab_feature_cl,polar_tab_feature_sigma2,
        polar_tab_feature_pref0,polar_tab_feature_pref1,
        polar_si_overlap,polar_field_feature_norms;
    Kokkos::View<int*> polar_output_permutation,polar_si_select;
    int polar_recursion_steps=0;
    // --- P3 模块⑤: 更新块(每个 SCF 步一个实例) ---
    // 依据: field_blocks.py::AgnosticEmbeddedOneBodyVariableUpdate L470-594;
    // uvu 路径表语义镜像 SparseUvuTensorProduct._path_meta L368-403。
    struct PolarUpdate {
        E3LinearKokkosT<Precision> potential_linear,node_feats_linear,
            charge_embedding,source_embedding,readout_linear_1,
            readout_linear_mid,readout_linear_2;
        Kokkos::View<Precision*> uvu_dot_weight,uvu_tp_weight;
        Kokkos::View<Precision*> uvu_dot_mask,uvu_tp_mask;
        Kokkos::View<int**,Kokkos::LayoutRight> uvu_dot_meta,
            uvu_tp_meta;                                  // (np,8)
        Kokkos::View<Precision*> uvu_dot_path_weight,uvu_tp_path_weight;
        Precision gate_scalar_cst=Precision(1),gate_gate_cst=Precision(1);
        int gate_scalar_count=0,gate_gate_count=0,gate_gated_count=0;
        Kokkos::View<Precision**,Kokkos::LayoutRight> nl_w0,nl_w1,nl_w2,nl_w3;
        Kokkos::View<Precision*> nl_b0,nl_b1,nl_b2,nl_b3;
        Kokkos::View<Precision*> ln0_weight,ln0_bias,ln1_weight,ln1_bias,
            ln2_weight,ln2_bias;
        Precision ln_eps=Precision(1e-5);
        int nl_in_width=0,nl_hidden_width=0;
        explicit PolarUpdate(const nlohmann::json& data);
        void set_e3_linear_backend(const std::string& backend);
        void set_e3_linear_workspace(
            const std::shared_ptr<typename E3LinearKokkosT<Precision>::Workspace>& workspace);
    };
    std::vector<E3LinearKokkosT<Precision>> polar_mixer_linears;
    std::vector<PolarUpdate> polar_updates;
    // ===== [v2 适配] 模块⑥⑦ hpp 私有增补(表与配置; 仅本类成员函数消费) =====
    // [私有] ⑦ 表与配置(module7_patch_final.md §4.1b 原文):
    // --- P3 模块⑦: local_electron_energy 表与配置（ctor 解析） ---
    // 依据: field_blocks.py L652-752（前向链）+ L294-466（SparseUvu mode0/mode1）。
    bool polar_le_enabled=false,polar_le_add=false;
    E3LinearKokkosT<Precision> polar_le_linear_up_q,polar_le_linear_up_v;
    Kokkos::View<int**,Kokkos::LayoutRight> polar_le_q_meta,
        polar_le_v_meta;                                // (np,8)
    Kokkos::View<Precision*> polar_le_q_path_weight,polar_le_q_weight,
        polar_le_q_mask,polar_le_v_path_weight,polar_le_v_weight,
        polar_le_v_mask;
    AffineMLPKokkosT<Precision> polar_le_mlp;   // RadialMLP 1024→128×3→1
    std::vector<LayerState> states;
    std::vector<ExecutionMH1NodeRuntime> execution_mh1_node_runtime;
    void prepare_execution_mh1_node_runtime_parameters();
    void reserve_execution_mh1_node_arena(int num_nodes);
    symmetrix::execution::MH1HostNodeProgramExpectation
        execution_mh1_host_node_expectation(int layer) const;
    symmetrix::execution::MH1CudaNodeProgramExpectation
        execution_mh1_cuda_node_expectation(int layer) const;
    ZBLKokkos zbl;
    std::unique_ptr<SphericalHarmonicsState> spherical_harmonics_state;
};

using MaceNonlinearKokkos = MaceNonlinearKokkosT<double>;
using MaceNonlinearFloatKokkos = MaceNonlinearKokkosT<float>;
