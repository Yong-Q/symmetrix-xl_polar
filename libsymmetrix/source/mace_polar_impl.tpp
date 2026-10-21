// mace_polar_impl.tpp — P3 极性静电场独立编译单元(2026-10-15)。
// 目的: polar 逻辑(②Fukui/③k网格+GTO/④势特征/⑤SCF; 及未来⑥⑦⑧)与主 tpp 分离,
//       之后 polar 改动只重编本 TU(单 TU ~数分钟)而非四个大 TU(20-40 分钟)。
// 由 mace_polar_kokkos.cpp 单独 include; 不要被其它 TU include。
// 各处依据/锚点沿用主 tpp 注释(未搬动的①/mixer/ctor 解析仍在主 tpp)。

#include "mace_nonlinear_kokkos.hpp"
#include "tools_kokkos.hpp"

#include <cstdlib>  // [dbg] std::getenv/atoi(POLAR_DBG_SKIP 通道隔离)

namespace {

template<typename Value>
std::vector<Value> tensor_values(const nlohmann::json& value){return value.at("values").get<std::vector<Value>>();}
const nlohmann::json& checked_interaction(const nlohmann::json& data){if(data.at("class").get<std::string>()!="RealAgnosticResidualNonLinearInteractionBlock")throw std::invalid_argument("MACE_Nonlinear Kokkos interaction class is unsupported.");return data;}
const nlohmann::json& readout_component(const nlohmann::json& data,int component){const auto readout_class=data.at("class").get<std::string>();if(readout_class=="LinearReadoutBlock")return data.at("linear");if(readout_class=="NonLinearReadoutBlock")return data.at(component==2?"linear_2":"linear_1");if(readout_class=="NonLinearBiasReadoutBlock"){if(component==3)return data.at("linear_mid");return data.at(component==2?"linear_2":"linear_1");}throw std::invalid_argument("MACE_Nonlinear Kokkos readout class is unsupported: "+readout_class);}

// P3 模块⑤: sparse_uvu 路径表解析(镜像 field_blocks.py SparseUvuTensorProduct
// L368-403 的 _path_meta 语义: 每条指令 = (in1_off,in2_off,out_off,w_off,
// mul1,mul2,mode,d1); mode0=l×l→0(成对收缩/√d1), mode1=l×0→l(标量调制)。
template<typename Precision>
void parse_uvu_paths(const nlohmann::json& data,std::vector<int>& meta,
    std::vector<Precision>& path_weights,std::vector<Precision>& weight_values,
    std::vector<Precision>& mask_values)
{
    const Irreps in1(data.at("irreps_in1").get<std::string>());
    const Irreps in2(data.at("irreps_in2").get<std::string>());
    const Irreps out(data.at("irreps_out").get<std::string>());
    weight_values=tensor_values<Precision>(data.at("weight"));
    mask_values=tensor_values<Precision>(data.at("output_mask"));
    int weight_offset=0;
    for(const auto& value:data.at("instructions")) {
        const auto& a=in1.blocks.at(value.at("i_in1").get<int>());
        const auto& b=in2.blocks.at(value.at("i_in2").get<int>());
        const auto& c=out.blocks.at(value.at("i_out").get<int>());
        const auto shape=value.at("path_shape").get<std::vector<int>>();
        const int d1=2*a.l+1;
        int mode=0;
        if(c.l!=0) {
            mode=1;
            if(b.l!=0||a.l!=c.l)
                throw std::invalid_argument(
                    "polar uvu: mode1 requires l2==0 and l1==l_out.");
        } else if(a.l!=b.l)
            throw std::invalid_argument(
                "polar uvu: mode0 requires l1==l2.");
        meta.insert(meta.end(),{a.offset,b.offset,c.offset,weight_offset,
            a.multiplicity,b.multiplicity,mode,d1});
        path_weights.push_back(value.at("path_weight").get<Precision>());
        weight_offset+=static_cast<int>(shape[0])*static_cast<int>(shape[1]);
    }
    if(weight_offset!=static_cast<int>(weight_values.size()))
        throw std::invalid_argument("polar uvu: weight size mismatch.");
}

template<class View>
void ensure_view(View& view,View& storage,std::size_t extent)
{
    if(storage.extent(0)<extent)
        Kokkos::realloc(Kokkos::WithoutInitializing,storage,extent);
    view=Kokkos::subview(
        storage,std::make_pair(std::size_t(0),extent));
}

template<class View>
void ensure_view(View& view,View& storage,int first,int second)
{
    if(storage.extent(0)<static_cast<std::size_t>(first)
        ||storage.extent(1)!=static_cast<std::size_t>(second))
        Kokkos::realloc(Kokkos::WithoutInitializing,storage,first,second);
    view=Kokkos::subview(storage,std::make_pair(0,first),Kokkos::ALL);
}

template<class View>
void ensure_view(View& view,View& storage,int first,int second,int third)
{
    // Do not subview the innermost dimension: a sliced LayoutRight view
    // deduces LayoutStride and cannot be assigned back to a LayoutRight
    // View.  Grow-only allocation keeps the layout identical; callers only
    // index columns below their logical extent.
    if(storage.extent(0)!=static_cast<std::size_t>(first)
        ||storage.extent(1)!=static_cast<std::size_t>(second)
        ||storage.extent(2)<static_cast<std::size_t>(third))
        Kokkos::realloc(Kokkos::WithoutInitializing,storage,
            first,second,third);
    view=storage;
}

// Plan B1: MATOS_REPLAY_CACHE=1 caches the forward SCF step state and feeds
// it to the reverse; 0 (default) keeps the replay path bit-for-bit.
inline bool polar_replay_cache_enabled()
{
    const char* value=std::getenv("MATOS_REPLAY_CACHE");
    return value!=nullptr&&std::atoi(value)!=0;
}

// UVU team tiling.  The uvu inner products stream one node's x1/x2 vector row
// per (node,u) or (node,v) thread; when tile_m divides a warp exactly, a warp
// owns one node row and the row load becomes a warp-wide broadcast instead of
// a 4-sector gather, and the adjoint/weight tile accesses stay coalesced.
// Both knobs are env-overridable for tuning; the defaults are the measured
// optimum.  Team size is capped at the CUDA limit (1024).
inline void uvu_tile_shape(int& tile_n,int& tile_m)
{
    int n=32,m=16;
    if(const char* e=std::getenv("MATOS_UVU_TILE_N")) {
        const int value=std::atoi(e);
        if(value>0&&value<=1024) n=value;
    }
    if(const char* e=std::getenv("MATOS_UVU_TILE_M")) {
        const int value=std::atoi(e);
        if(value>0&&value<=1024) m=value;
    }
    while(n*m>1024) {
        if(m>=n) m/=2; else n/=2;
    }
    tile_n=n;
    tile_m=m;
}

// P3 模块⑤: sparse_uvu 路径表解析(镜像 field_blocks.py SparseUvuTensorProduct
// L368-403 的 _path_meta 语义: 每条指令 = (in1_off,in2_off,out_off,w_off,
// mul1,mul2,mode,d1); mode0=l×l→0(成对收缩/√d1), mode1=l×0→l(标量调制)。

}  // namespace

template<typename Precision>
MaceNonlinearKokkosT<Precision>::PolarUpdate::PolarUpdate(
    const nlohmann::json& data)
    :potential_linear(data.at("potential_embedding").at("potential_linear")),
     node_feats_linear(data.at("potential_embedding").at("node_feats_linear")),
     charge_embedding(data.at("potential_embedding").at("charge_embedding")),
     source_embedding(data.at("source_embedding")),
     readout_linear_1(data.at("readout").at("linear_1")),
     readout_linear_mid(data.at("readout").at("linear_mid")),
     readout_linear_2(data.at("readout").at("linear_2"))
{
    // uvu 路径表(依据 field_blocks.py L368-403; path_weight 经审计补序列化)
    {
        std::vector<int> meta;
        std::vector<Precision> path_weights,weight_values,mask_values;
        parse_uvu_paths(data.at("dot_products"),meta,path_weights,
            weight_values,mask_values);
        const int dot_paths=static_cast<int>(path_weights.size());
        uvu_dot_meta=toKokkosView("polar uvu dot meta",meta,dot_paths,8);
        uvu_dot_path_weight=toKokkosView(
            "polar uvu dot path weight",path_weights);
        uvu_dot_weight=toKokkosView("polar uvu dot weight",weight_values);
        uvu_dot_mask=toKokkosView("polar uvu dot mask",mask_values);
        meta.clear();path_weights.clear();weight_values.clear();mask_values.clear();
        parse_uvu_paths(data.at("tp_out"),meta,path_weights,
            weight_values,mask_values);
        const int tp_paths=static_cast<int>(path_weights.size());
        uvu_tp_meta=toKokkosView("polar uvu tp meta",meta,tp_paths,8);
        uvu_tp_path_weight=toKokkosView(
            "polar uvu tp path weight",path_weights);
        uvu_tp_weight=toKokkosView("polar uvu tp weight",weight_values);
        uvu_tp_mask=toKokkosView("polar uvu tp mask",mask_values);
    }
    // e3nn.nn.Gate(readout 中段; 语义经 charges_hooks 数值锁定 2.2e-16:
    // scalars=cst_s·silu; gates=cst_g·sigmoid; vec=cst_g·sigmoid⊗gated)
    {
        const auto& gate=data.at("readout_gate");
        gate_scalar_cst=gate.at("act_scalars").at(0).at("cst")
            .get<Precision>();
        gate_gate_cst=gate.at("act_gates").at(0).at("cst")
            .get<Precision>();
        // 分组计数(irreps 形如 "32x0e");切片 [scalars|gates|gated(3×)]。
        const auto parse_count=[&](const char* key) {
            const std::string spec=gate.at(key).get<std::string>();
            const auto separator=spec.find('x');
            return separator==std::string::npos
                ?0:std::stoi(spec.substr(0,separator));
        };
        gate_scalar_count=parse_count("irreps_scalars");
        gate_gate_count=parse_count("irreps_gates");
        gate_gated_count=parse_count("irreps_gated");
    }
    // RadialMLP: [linear, LayerNorm, silu]×3 + linear (mace.modules.radial FullyConnectedNet)
    {
        const auto& layers=data.at("nonlinearity").at("layers");
        const auto dense_weight=[&](const nlohmann::json& layer,
                const char* label) {
            const auto weight=tensor_values<Precision>(layer.at("weight"));
            const auto shape=layer.at("weight").at("shape")
                .get<std::vector<int>>();
            return toKokkosView(label,weight,shape.at(0),shape.at(1));
        };
        nl_w0=dense_weight(layers.at(0),"polar nl w0");
        nl_w1=dense_weight(layers.at(3),"polar nl w1");
        nl_w2=dense_weight(layers.at(6),"polar nl w2");
        nl_w3=dense_weight(layers.at(9),"polar nl w3");
        nl_b0=toKokkosView("polar nl b0",
            tensor_values<Precision>(layers.at(0).at("bias")));
        nl_b1=toKokkosView("polar nl b1",
            tensor_values<Precision>(layers.at(3).at("bias")));
        nl_b2=toKokkosView("polar nl b2",
            tensor_values<Precision>(layers.at(6).at("bias")));
        nl_b3=toKokkosView("polar nl b3",
            tensor_values<Precision>(layers.at(9).at("bias")));
        ln0_weight=toKokkosView("polar ln0 weight",
            tensor_values<Precision>(layers.at(1).at("weight")));
        ln0_bias=toKokkosView("polar ln0 bias",
            tensor_values<Precision>(layers.at(1).at("bias")));
        ln1_weight=toKokkosView("polar ln1 weight",
            tensor_values<Precision>(layers.at(4).at("weight")));
        ln1_bias=toKokkosView("polar ln1 bias",
            tensor_values<Precision>(layers.at(4).at("bias")));
        ln2_weight=toKokkosView("polar ln2 weight",
            tensor_values<Precision>(layers.at(7).at("weight")));
        ln2_bias=toKokkosView("polar ln2 bias",
            tensor_values<Precision>(layers.at(7).at("bias")));
        ln_eps=layers.at(1).at("eps").get<Precision>();
        const auto input_shape=layers.at(0).at("weight").at("shape")
            .get<std::vector<int>>();
        nl_in_width=input_shape.at(1);
        nl_hidden_width=input_shape.at(0);
    }
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::PolarUpdate::set_e3_linear_backend(
    const std::string& backend)
{
    potential_linear.set_backend(backend);
    node_feats_linear.set_backend(backend);
    charge_embedding.set_backend(backend);
    source_embedding.set_backend(backend);
    readout_linear_1.set_backend(backend);
    readout_linear_mid.set_backend(backend);
    readout_linear_2.set_backend(backend);
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::PolarUpdate::set_e3_linear_workspace(
    const std::shared_ptr<typename E3LinearKokkosT<Precision>::Workspace>&
        workspace)
{
    potential_linear.set_workspace(workspace);
    node_feats_linear.set_workspace(workspace);
    charge_embedding.set_workspace(workspace);
    source_embedding.set_workspace(workspace);
    readout_linear_1.set_workspace(workspace);
    readout_linear_mid.set_workspace(workspace);
    readout_linear_2.set_workspace(workspace);
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::run_polar_pipeline(int num_nodes,
    Kokkos::View<Precision**,Kokkos::LayoutRight> node_attrs,
    Kokkos::View<const double*> node_positions,
    Kokkos::View<const double*> cell,
    double total_charge,double total_spin)
{
    symmetrix_polar_profile_mark("pipeline:enter");
    polar_rev_cache_valid=false;   // Plan B1: invalidate any previous-eval cache
    const int owned_nodes=polar_owned_node_count(num_nodes);
    // 工作密度: ②/⑤ 在其上改写; ① 的原始累积(polar_spin_charge_density)
    // 保持只读(修复 ① 门语义漂移: 该视图此前被 ②/⑤ 原地改写)。
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_working_density(
        "polar working density",num_nodes,
        static_cast<int>(polar_spin_charge_density.extent(1)));
    ordered_kokkos_deep_copy(polar_working_density,polar_spin_charge_density);
    if(polar_enabled) {
        symmetrix_polar_profile_mark("stage2:fukui");
        // P3 模块②: Fukui 初始化校正(依据: mace extensions.py L737-766)。
        // fukui 源(全列 2 通道;输入=末层 node_feats 的 e3nn 基转换张量,
        // 标量通道与内部基一致, 转换保持口径统一)。
        if(static_cast<int>(polar_fukui_source_map->linear_2.output_dimension())!=2)
            throw std::runtime_error(
                "polar_blocks: fukui_source_map must output 2 channels.");
        ensure_view(polar_fukui_sources,polar_fukui_sources_storage,num_nodes,2);
        polar_fukui_source_map->evaluate_all(
            polar_node_feats_e3nn,polar_fukui_sources);
        auto polar_sources=polar_fukui_sources;
        auto polar_accumulator=polar_working_density;
        // 每图(本调用=单图,整盒)归约: 源和 + l=0 两通道预测总量。
        const auto polar_sources_host=Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),polar_sources);
        const auto polar_accumulator_host=Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),polar_accumulator);
        Precision source_sums[2]={Precision(0),Precision(0)};
        Precision predicted_zero[2]={Precision(0),Precision(0)};
        for(int node=0;node<owned_nodes;++node) {
            source_sums[0]+=polar_sources_host(node,0);
            source_sums[1]+=polar_sources_host(node,1);
            predicted_zero[0]+=polar_accumulator_host(node,0);
            predicted_zero[1]+=polar_accumulator_host(node,4);
        }
        {
            double global_sums[4]={static_cast<double>(source_sums[0]),
                static_cast<double>(source_sums[1]),
                static_cast<double>(predicted_zero[0]),
                static_cast<double>(predicted_zero[1])};
            polar_global_sum_host(global_sums,4);
            source_sums[0]=Precision(global_sums[0]);
            source_sums[1]=Precision(global_sums[1]);
            predicted_zero[0]=Precision(global_sums[2]);
            predicted_zero[1]=Precision(global_sums[3]);
        }
        const Precision raw_source_sums[2]={source_sums[0],source_sums[1]};
        for(int channel=0;channel<2;++channel)
            if(source_sums[channel]==Precision(0))
                source_sums[channel]=Precision(1);
        // Q_p_S=Q+(S-1), Q_m_S=Q-(S-1)(extensions.py L752-753);
        // total_spin 为 2S+1 多重度(mace info 口径, 默认 1)。
        const Precision charge=Precision(total_charge);
        const Precision spin_multiplicity=Precision(total_spin);
        const Precision target_alpha=Precision(0.5)
            *(charge+(spin_multiplicity-Precision(1)));
        const Precision target_beta=Precision(0.5)
            *(charge-(spin_multiplicity-Precision(1)));
        const Precision factor_alpha=(target_alpha-predicted_zero[0])
            /source_sums[0];
        const Precision factor_beta=(target_beta-predicted_zero[1])
            /source_sums[1];
        Kokkos::parallel_for("polar fukui correction",num_nodes,
            KOKKOS_LAMBDA(int node) {
                polar_accumulator(node,0)+=
                    polar_sources(node,0)*factor_alpha;
                polar_accumulator(node,4)+=
                    polar_sources(node,1)*factor_beta;
            });
        // P3 模块⑧: 初始化 Fukui 标量缓存([Sα,Sβ,predα,predβ,Fα,Fβ];S 为钳制前原始值)。
        polar_fukui_init_scalars={
            raw_source_sums[0],raw_source_sums[1],
            predicted_zero[0],predicted_zero[1],
            factor_alpha,factor_beta};
        // 校正后 = field_independent_spin_charge_density(mace L774 clone)。
        ensure_view(polar_field_independent_density,
            polar_field_independent_density_storage,num_nodes,
            static_cast<int>(polar_accumulator.extent(1)));
        ordered_kokkos_deep_copy(
            polar_field_independent_density,polar_working_density);
    }

    if(polar_enabled) {
        symmetrix_polar_profile_mark("stage3:kgeom");
        // ---- P3 模块③: k 网格 + GTO 几何预计算 ----
        // 依据: graph_longrange/kspace.py::compute_k_vectors_flat L11-127(半空间
        // 枚举+滤波+k0 掩码), gto_utils.py::GTOBasis L69-178(径向/cl_scale/相位),
        // features.py::_pbc_precompute_geometry L568-615(cos/sin/因子)。公式经
        // numpy 预检对 geo3 参考 dump 逐位锁定: rcell=2π·inv(cellᵀ) 0.0; k 0.0;
        // cos/sin 1.1e-16; basis 实部 0.0; l=1 项 1.1e-16(e3nn normalize=True 为
        // 输入向量归一化 → f1·Y1 = c1·pref1·cl1·exp(-½σ²k²)·v, k=0 处解析为 0)。
        if(node_positions.extent(0)!=static_cast<std::size_t>(3)*num_nodes
            ||cell.extent(0)!=9)
            throw std::runtime_error(
                "polar_blocks requires node positions (3N) and a 3x3 cell.");
        const auto cell_host=Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),cell);
        double lattice[3][3];
        for(int row=0;row<3;++row)
            for(int column=0;column<3;++column)
                lattice[row][column]=cell_host(3*row+column);
        constexpr double two_pi=6.283185307179586476925286766559;
        double reciprocal[3][3];
        {
            const double a=lattice[0][0],b=lattice[0][1],c=lattice[0][2];
            const double d=lattice[1][0],e=lattice[1][1],f=lattice[1][2];
            const double g=lattice[2][0],h=lattice[2][1],i=lattice[2][2];
            const double det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);
            const double inv=1.0/det;
            const double m00=(e*i-f*h)*inv,m01=(c*h-b*i)*inv,m02=(b*f-c*e)*inv;
            const double m10=(f*g-d*i)*inv,m11=(a*i-c*g)*inv,m12=(c*d-a*f)*inv;
            const double m20=(d*h-e*g)*inv,m21=(b*g-a*h)*inv,m22=(a*e-b*d)*inv;
            // rcell = 2π·inv(cellᵀ) → rc[i][j] = 2π·inv_cell[j][i]
            reciprocal[0][0]=two_pi*m00;reciprocal[0][1]=two_pi*m10;
            reciprocal[0][2]=two_pi*m20;
            reciprocal[1][0]=two_pi*m01;reciprocal[1][1]=two_pi*m11;
            reciprocal[1][2]=two_pi*m21;
            reciprocal[2][0]=two_pi*m02;reciprocal[2][1]=two_pi*m12;
            reciprocal[2][2]=two_pi*m22;
        }
        const double volume=std::abs(
            lattice[0][0]*(lattice[1][1]*lattice[2][2]
                -lattice[1][2]*lattice[2][1])
            -lattice[0][1]*(lattice[1][0]*lattice[2][2]
                -lattice[1][2]*lattice[2][0])
            +lattice[0][2]*(lattice[1][0]*lattice[2][1]
                -lattice[1][1]*lattice[2][0]));
        // P3 模块⑧: 缓存 volume/rcell 供反向(⑥ ρ̄ 的 V 因子 + 未来 cell 伴随,见 §6)。
        polar_cell_volume=volume;
        for(int row=0;row<3;++row)
            for(int column=0;column<3;++column)
                polar_rcell_host[3*row+column]=Precision(reciprocal[row][column]);
        // ---- P4 RBE 分支: 随机批采样(全空间; 逐轴取整高斯 + MH) ----
        // 依据 docs/design/RBE_P4_DESIGN.md §1.2(Box-Muller+Metropolis 提议)/
        // §3.3(采样包络 q(k)∝e^{−σ_q²k²/2}, σ_q=密度基宽 1.5; 1/k² 留在估计量)。
        // 采样空间 = m∈Z³\{0}, k = rcellᵀ·m;不做 |k|≤cutoff 截断(全支撑, §3.3)。
        // RNG = counter-based SplitMix64(纯函数于 (seed,frame,slot) ⇒ 与调度无关、
        // 跨运行逐位可复现;种子入 JSON,见 PLAN 依据账本)。首版限正交晶胞。
        std::vector<Precision> k_flat,k_norm2_host,k0_host;
        if(polar_rbe_enabled) {
            for(int i2=0;i2<3;++i2)
                for(int j2=0;j2<3;++j2)
                    if(i2!=j2&&std::fabs(lattice[i2][j2])
                        >1e-10*std::max(1.0,std::fabs(lattice[i2][i2])))
                        throw std::runtime_error(
                            "polar_blocks: RBE 首版仅支持正交晶胞"
                            " (off-diagonal cell entries must vanish).");
            double s_axis[3];
            for(int axis=0;axis<3;++axis) {
                const double length=std::sqrt(
                    lattice[axis][0]*lattice[axis][0]
                    +lattice[axis][1]*lattice[axis][1]
                    +lattice[axis][2]*lattice[axis][2]);
                s_axis[axis]=length/(two_pi*polar_rbe_sigma);
            }
            // Z_q = Σ_{m≠0} e^{−σ_q²k²/2} = Π_i S_i − 1(每轴精确直和,非 Poisson 近似)
            double zq=1.0;
            for(int axis=0;axis<3;++axis) {
                const double inv2s2=1.0/(2.0*s_axis[axis]*s_axis[axis]);
                const int m_max=std::max(
                    64,static_cast<int>(std::ceil(16.0*s_axis[axis])));
                double axis_sum=1.0;
                for(int m=1;m<=m_max;++m)
                    axis_sum+=2.0*std::exp(
                        -static_cast<double>(m)*static_cast<double>(m)*inv2s2);
                zq*=axis_sum;
            }
            zq-=1.0;
            polar_rbe_zq=zq;
            const unsigned long long frame_id=static_cast<unsigned long long>(
                polar_rbe_frame);
            ++polar_rbe_frame;
            const int p=polar_rbe_p;
            const unsigned long long seed=polar_rbe_seed;
            const double s0=s_axis[0],s1=s_axis[1],s2=s_axis[2];
            const double r00=reciprocal[0][0],r10=reciprocal[1][0],
                r20=reciprocal[2][0];
            const double r01=reciprocal[0][1],r11=reciprocal[1][1],
                r21=reciprocal[2][1];
            const double r02=reciprocal[0][2],r12=reciprocal[1][2],
                r22=reciprocal[2][2];
            Kokkos::View<Precision*> m_view("polar rbe m",std::size_t(3)*p);
            auto m_dev=m_view;
            Kokkos::parallel_for("polar rbe sample",1,KOKKOS_LAMBDA(int) {
                const auto splitmix=[](unsigned long long x) {
                    x+=0x9E3779B97F4A7C15ULL;
                    x=(x^(x>>30))*0xBF58476D1CE4E5B9ULL;
                    x=(x^(x>>27))*0x94D049BB133111EBULL;
                    return x^(x>>31);
                };
                unsigned long long ctr=1ULL;
                const auto next_uniform=[&](unsigned long long idx)->double {
                    const unsigned long long v=splitmix(seed
                        +idx*0x100000001B3ULL
                        +frame_id*0x9E3779B97F4A7C15ULL);
                    return (static_cast<double>(v>>11)+0.5)
                        *(1.0/9007199254740992.0); // (0,1)
                };
                const auto norm_z=[&]()->double {
                    double u1=next_uniform(ctr);++ctr;
                    const double u2=next_uniform(ctr);++ctr;
                    if(u1<1e-300) u1=1e-300;
                    return Kokkos::sqrt(-2.0*Kokkos::log(u1))
                        *Kokkos::cos(6.283185307179586476925286766559*u2);
                };
                const auto erf_a=[](double x)->double {
                    const double sgn=x<0.0?-1.0:1.0;
                    if(x<0.0) x=-x;
                    const double t=1.0/(1.0+0.3275911*x);
                    const double y=1.0-((((1.061405429*t-1.453152027)*t
                        +1.421413741)*t-0.284496736)*t+0.254829592)*t
                        *Kokkos::exp(-x*x);
                    return sgn*y;
                };
                const auto phi=[&](double z)->double {
                    return 0.5*(1.0+erf_a(z*0.70710678118654752440084436210485));
                };
                const auto q_mass=[&](int m,double s)->double {
                    return phi((static_cast<double>(m)+0.5)/s)
                        -phi((static_cast<double>(m)-0.5)/s);
                };
                double mx[3]={0.0,0.0,0.0};
                for(int slot=0;slot<p;++slot) {
                    for(;;) {
                        double cand[3];
                        for(int axis=0;axis<3;++axis) {
                            const double s=axis==0?s0:(axis==1?s1:s2);
                            const int m_new=static_cast<int>(
                                Kokkos::round(s*norm_z()));
                            if(slot==0) {
                                cand[axis]=static_cast<double>(m_new);
                                continue;
                            }
                            // MH(独立提议; 目标 p∝e^{−m²/2s²}; 提议 q=区间质量,
                            // 同 lammps-rbe kspace_style rbe L77-95/L299 语义)。
                            const double p_old=Kokkos::exp(
                                -0.5*mx[axis]*mx[axis]/(s*s));
                            const double p_new=Kokkos::exp(
                                -0.5*static_cast<double>(m_new)
                                *static_cast<double>(m_new)/(s*s));
                            const double q_old=q_mass(
                                static_cast<int>(mx[axis]),s);
                            const double q_new=q_mass(m_new,s);
                            double acce=(p_new*q_old)/(p_old*q_new);
                            if(acce>1.0) acce=1.0;
                            const double u=next_uniform(ctr);++ctr;
                            cand[axis]=(u<acce)
                                ?static_cast<double>(m_new):mx[axis];
                        }
                        if(cand[0]==0.0&&cand[1]==0.0&&cand[2]==0.0)
                            continue; // m≠0(采样空间)
                        for(int axis=0;axis<3;++axis) mx[axis]=cand[axis];
                        break;
                    }
                    m_dev(3*slot)=Precision(mx[0]);
                    m_dev(3*slot+1)=Precision(mx[1]);
                    m_dev(3*slot+2)=Precision(mx[2]);
                }
            });
            const auto m_host=Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(),m_view);
            for(int slot=0;slot<p;++slot) {
                const double n1=m_host(3*slot),n2=m_host(3*slot+1),
                    n3=m_host(3*slot+2);
                const double kx=n1*r00+n2*r10+n3*r20;
                const double ky=n1*r01+n2*r11+n3*r21;
                const double kz=n1*r02+n2*r12+n3*r22;
                k_flat.push_back(Precision(kx));
                k_flat.push_back(Precision(ky));
                k_flat.push_back(Precision(kz));
                k_norm2_host.push_back(Precision(kx*kx+ky*ky+kz*kz));
                k0_host.push_back(Precision(0));
            }
        } else {
        // n 上界(kspace.py L41-53): max_ns=ceil(cutoff / (rcell_i·ĉell_i))
        int max_ns[3];
        for(int axis=0;axis<3;++axis) {
            const double norm=std::sqrt(lattice[axis][0]*lattice[axis][0]
                +lattice[axis][1]*lattice[axis][1]
                +lattice[axis][2]*lattice[axis][2]);
            const double dot=(reciprocal[axis][0]*lattice[axis][0]
                +reciprocal[axis][1]*lattice[axis][1]
                +reciprocal[axis][2]*lattice[axis][2])/norm;
            max_ns[axis]=static_cast<int>(
                std::ceil(polar_kspace_cutoff/dot));
        }
        // 半空间枚举(torch.cartesian_prod 末维最快; origin 居首)
        struct LatticePoint { int n1,n2,n3; };
        std::vector<LatticePoint> lattice_points;
        lattice_points.push_back({0,0,0});
        for(int n3=1;n3<max_ns[2];++n3)
            lattice_points.push_back({0,0,n3});
        for(int n2=1;n2<max_ns[1];++n2)
            for(int n3=-max_ns[2];n3<max_ns[2];++n3)
                lattice_points.push_back({0,n2,n3});
        for(int n1=1;n1<max_ns[0];++n1)
            for(int n2=-max_ns[1];n2<max_ns[1];++n2)
                for(int n3=-max_ns[2];n3<max_ns[2];++n3)
                    lattice_points.push_back({n1,n2,n3});
        for(const auto& n:lattice_points) {
            const double kx=n.n1*reciprocal[0][0]+n.n2*reciprocal[1][0]
                +n.n3*reciprocal[2][0];
            const double ky=n.n1*reciprocal[0][1]+n.n2*reciprocal[1][1]
                +n.n3*reciprocal[2][1];
            const double kz=n.n1*reciprocal[0][2]+n.n2*reciprocal[1][2]
                +n.n3*reciprocal[2][2];
            const double norm2=kx*kx+ky*ky+kz*kz;
            if(norm2>static_cast<double>(polar_kspace_cutoff)
                *static_cast<double>(polar_kspace_cutoff)) continue;
            k_flat.push_back(Precision(kx));
            k_flat.push_back(Precision(ky));
            k_flat.push_back(Precision(kz));
            k_norm2_host.push_back(Precision(norm2));
            k0_host.push_back(k_norm2_host.size()==1
                ?Precision(1):Precision(0));
        }
        } // (else: 确定性半空间枚举)
        const int num_k=static_cast<int>(k_norm2_host.size());
        polar_k_vectors=toKokkosView("polar k vectors",k_flat,num_k,3);
        polar_k_norm2=toKokkosView("polar k norm2",k_norm2_host);
        polar_k0_mask=toKokkosView("polar k0 mask",k0_host);
        // cos/sin(k·r)(features.py L579-583; 单图=全矩阵)
        Kokkos::realloc(Kokkos::WithoutInitializing,
            polar_cosines,num_k,num_nodes);
        Kokkos::realloc(Kokkos::WithoutInitializing,
            polar_sines,num_k,num_nodes);
        auto kvecs=polar_k_vectors;
        auto kcos=polar_cosines;
        auto ksin=polar_sines;
        auto positions=node_positions;
        Kokkos::parallel_for("polar k inner products",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{num_k,num_nodes}),
            KOKKOS_LAMBDA(int k,int node) {
                const Precision inner=kvecs(k,0)*Precision(positions(3*node))
                    +kvecs(k,1)*Precision(positions(3*node+1))
                    +kvecs(k,2)*Precision(positions(3*node+2));
                kcos(k,node)=Kokkos::cos(inner);
                ksin(k,node)=Kokkos::sin(inner);
            });
        // GTO 基函数(实/虚傅里叶系数; 表全部来自 ctor 解析)
        Kokkos::realloc(Kokkos::WithoutInitializing,
            polar_density_basis_re,num_k,4);
        Kokkos::realloc(Kokkos::WithoutInitializing,
            polar_density_basis_im,num_k,4);
        Kokkos::realloc(Kokkos::WithoutInitializing,
            polar_feature_basis_re,num_k,8);
        Kokkos::realloc(Kokkos::WithoutInitializing,
            polar_feature_basis_im,num_k,8);
        {
            auto dbre=polar_density_basis_re,dbim=polar_density_basis_im;
            auto fbre=polar_feature_basis_re,fbim=polar_feature_basis_im;
            auto d_cl=polar_tab_density_cl,d_s2=polar_tab_density_sigma2;
            auto d_p0=polar_tab_density_pref0,d_p1=polar_tab_density_pref1;
            auto f_cl=polar_tab_feature_cl,f_s2=polar_tab_feature_sigma2;
            auto f_p0=polar_tab_feature_pref0,f_p1=polar_tab_feature_pref1;
            auto kn2=polar_k_norm2;
            auto k0m=polar_k0_mask;
            auto kvec=polar_k_vectors;
            const Precision y0=Precision(0.28209479177387814);
            const Precision y1=Precision(0.48860251190291992);
            const int num_sigmas=polar_num_feature_sigmas;
            Kokkos::parallel_for("polar gto basis",num_k,
                KOKKOS_LAMBDA(int k) {
                    const Precision norm2=kn2(k);
                    const Precision vx=kvec(k,1),vy=kvec(k,2),vz=kvec(k,0);
                    const Precision exp_d=Kokkos::exp(
                        Precision(-0.5)*norm2*d_s2(0));
                    dbre(k,0)=d_p0(0)*d_cl(0)*y0*exp_d;
                    dbim(k,0)=Precision(0);
                    const Precision cd=y1*d_p1(0)*d_cl(1)*exp_d;
                    dbre(k,1)=Precision(0);dbre(k,2)=Precision(0);
                    dbre(k,3)=Precision(0);
                    dbim(k,1)=-cd*vx;dbim(k,2)=-cd*vy;dbim(k,3)=-cd*vz;
                    for(int s=0;s<num_sigmas;++s) {
                        const Precision exp_f=Kokkos::exp(
                            Precision(-0.5)*norm2*f_s2(s));
                        const int base=4*s;
                        fbre(k,base)=f_p0(s)*f_cl(2*s)*y0*exp_f;
                        fbim(k,base)=Precision(0);
                        const Precision cf=y1*f_p1(s)*f_cl(2*s+1)*exp_f;
                        fbre(k,base+1)=Precision(0);fbre(k,base+2)=Precision(0);
                        fbre(k,base+3)=Precision(0);
                        fbim(k,base+1)=-cf*vx;fbim(k,base+2)=-cf*vy;
                        fbim(k,base+3)=-cf*vz;
                    }
                });
        }
        // ---- P4 RBE: 每 k 估计量权重 ----
        // 确定性=1(乘 1.0 逐位精确 ⇒ 既有路径零改动);RBE=Z_q·e^{+σ_q²k²/2}/(2P),
        // 1/2 为半空间↔全空间换算(RBE_P4_DESIGN §3.3 L2/L3)。乘点=各 k 求和项
        // (投影累加/⑥ 能量/⑥ ρ̄/反投 ii、iii-b);assemble 与相位伴随经 ρ̄ 携带,不再单乘。
        ensure_view(polar_k_weight,polar_k_weight_storage,
            std::size_t(num_k));
        {
            auto kw=polar_k_weight;
            if(polar_rbe_enabled) {
                const Precision rbe_scale=Precision(polar_rbe_zq
                    /(2.0*static_cast<double>(polar_rbe_p)));
                const Precision rbe_half_sigma2=Precision(
                    0.5*polar_rbe_sigma*polar_rbe_sigma);
                auto kw_kn2=polar_k_norm2;
                Kokkos::parallel_for("polar rbe weights",num_k,
                    KOKKOS_LAMBDA(int k) {
                        kw(k)=rbe_scale*Kokkos::exp(
                            rbe_half_sigma2*kw_kn2(k));
                    });
            } else {
                Kokkos::parallel_for("polar det weights",num_k,
                    KOKKOS_LAMBDA(int k) {
                        kw(k)=Precision(1);
                    });
            }
        }
        symmetrix_polar_profile_mark("stage4:potfeat");
        // ---- P3 模块④: forward_dynamic(逐通道势特征) ----
        // 依据: features.py L617-658(assemble→库仑核→投影→permute→减去自相互作用)
        // + extensions.py L834-838(逐通道归一)与 L846-848(α|β 拼接)。
        Kokkos::realloc(Kokkos::WithoutInitializing,
            polar_potential_features,num_nodes,16);
        {
            constexpr double field_constant=1.0/(5.526349406e-3);
            const double two_pi_cubed=two_pi*two_pi*two_pi;
            const Precision rho_scale=Precision(two_pi_cubed/volume);
            const Precision proj_scale=Precision(2.0/two_pi_cubed);
            Kokkos::View<Precision*> rho_re("polar rho re",num_k);
            Kokkos::View<Precision*> rho_im("polar rho im",num_k);
            auto kcos2=polar_cosines;
            auto ksin2=polar_sines;
            auto dbre=polar_density_basis_re,dbim=polar_density_basis_im;
            auto fbre=polar_feature_basis_re,fbim=polar_feature_basis_im;
            auto kn2=polar_k_norm2,k0m=polar_k0_mask;
            auto knorm=polar_field_feature_norms;
            auto si_overlap=polar_si_overlap;
            auto si_select=polar_si_select;
            auto permutation=polar_output_permutation;
            auto density_all=polar_working_density;
            auto potential_features=polar_potential_features;
            auto kweight=polar_k_weight;
            for(int channel=0;channel<2;++channel) {
                // assemble: rho(k) = (2π)³/V · [Bᵣ·Σₙ C·q + Bᵢ·Σₙ S·q,
                //                               Bᵢ·Σₙ C·q − Bᵣ·Σₙ S·q]
                Kokkos::parallel_for("polar assemble fourier",num_k,
                    KOKKOS_LAMBDA(int k) {
                        Precision acc_cos[4],acc_sin[4];
                        for(int m=0;m<4;++m) {
                            acc_cos[m]=Precision(0);acc_sin[m]=Precision(0);
                        }
                        for(int node=0;node<owned_nodes;++node) {
                            const Precision c=kcos2(k,node);
                            const Precision s=ksin2(k,node);
                            for(int m=0;m<4;++m) {
                                const Precision q=density_all(
                                    node,4*channel+m);
                                acc_cos[m]+=c*q;
                                acc_sin[m]+=s*q;
                            }
                        }
                        Precision re=Precision(0),im=Precision(0);
                        for(int m=0;m<4;++m) {
                            re+=dbre(k,m)*acc_cos[m]+dbim(k,m)*acc_sin[m];
                            im+=dbim(k,m)*acc_cos[m]-dbre(k,m)*acc_sin[m];
                        }
                        rho_re(k)=rho_scale*re;
                        rho_im(k)=rho_scale*im;
                    });
                polar_global_sum_view(rho_re);
                polar_global_sum_view(rho_im);
                // project: features[n,·](8) = permute(2Σₖ (A·C+B·S)/(2π)³),
                // 再减自相互作用(逐列常数×选取分量)与逐通道归一。
                // One thread per (node, output j).  j uses only feat[perm(j)],
                // so the k accumulation stays in the original ascending order
                // (bitwise-identical FP64) while exposing 8x more threads.
                Kokkos::parallel_for("polar project features",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                        {num_nodes,8}),
                    KOKKOS_LAMBDA(int node,int j) {
                        const int m=permutation(j);
                        Precision proj_cos=Precision(0);
                        Precision proj_sin=Precision(0);
                        for(int k=0;k<num_k;++k) {
                            const Precision inv_k2=(k0m(k)>Precision(0))
                                ?Precision(0)
                                :Precision(field_constant)/kn2(k);
                            const Precision pr=rho_re(k)*inv_k2;
                            const Precision pi=rho_im(k)*inv_k2;
                            const Precision kfp=(k0m(k)>Precision(0))
                                ?Precision(0.5):Precision(1);
                            const Precision fa=(pr*fbre(k,m)
                                +pi*fbim(k,m))*kfp;
                            const Precision fb=(pr*fbim(k,m)
                                -pi*fbre(k,m))*kfp;
                            proj_cos+=(kweight(k)*fa)*kcos2(k,node);
                            proj_sin+=(kweight(k)*fb)*ksin2(k,node);
                        }
                        const Precision value=proj_scale
                            *(proj_cos+proj_sin)
                            -si_overlap(j)*density_all(
                                node,4*channel+si_select(j));
                        potential_features(node,8*channel+j)=value/knorm(j);
                    });
            }
        }

        symmetrix_polar_profile_mark("stage5:scf");
        // ---- P3 模块⑤: SCF 循环(依据 extensions.py L777-866) ----
        // 每步: 逐自旋通道描述符(α/β) → 归一拼接(16) → 更新块(势嵌入三分量 /
        // uvu 点积 / source 嵌入 / MLP(LN+silu) / uvu tp / 双重 Gate 读出) →
        // 密度 += out[:,:8] → 逐步 Fukui 重归一+l=0 校正(out[:,8:10])。
        // 中间量访问器 = 最后一次迭代值(与 charges_hooks step1 对齐)。
        {
            constexpr double scf_field_constant=1.0/(5.526349406e-3);
            const double scf_two_pi_cubed=two_pi*two_pi*two_pi;
            const Precision scf_rho_scale=Precision(scf_two_pi_cubed/volume);
            const Precision scf_proj_scale=Precision(2.0/scf_two_pi_cubed);
            Kokkos::View<Precision*> step_rho_re("polar scf rho re",num_k);
            Kokkos::View<Precision*> step_rho_im("polar scf rho im",num_k);
            Kokkos::realloc(Kokkos::WithoutInitializing,
                polar_scf_step_densities,polar_recursion_steps,num_nodes,8);
            Kokkos::realloc(Kokkos::WithoutInitializing,
                polar_field_feats_a,num_nodes,8);
            Kokkos::realloc(Kokkos::WithoutInitializing,
                polar_field_feats_b,num_nodes,8);
            Kokkos::realloc(Kokkos::WithoutInitializing,
                polar_step_potential_features,num_nodes,16);
            Kokkos::realloc(Kokkos::WithoutInitializing,
                polar_pe_out,num_nodes,
                polar_updates.front().potential_linear.output_dimension());
            ensure_view(polar_scratch_a,polar_scratch_a_storage,num_nodes,
                polar_updates.front().potential_linear.output_dimension());
            ensure_view(polar_scratch_b,polar_scratch_b_storage,num_nodes,
                polar_updates.front().potential_linear.output_dimension());
            const int dp_width=static_cast<int>(
                polar_updates.front().source_embedding.output_dimension());
            Kokkos::realloc(Kokkos::WithoutInitializing,
                polar_dp_out,num_nodes,dp_width);
            Kokkos::realloc(Kokkos::WithoutInitializing,
                polar_se_out,num_nodes,dp_width);
            ensure_view(polar_nl_input,polar_nl_input_storage,num_nodes,
                2*dp_width);
            ensure_view(polar_nl_a,polar_nl_a_storage,num_nodes,
                polar_updates.front().nl_hidden_width);
            ensure_view(polar_nl_b,polar_nl_b_storage,num_nodes,
                polar_updates.front().nl_hidden_width);
            Kokkos::realloc(Kokkos::WithoutInitializing,
                polar_nl_out,num_nodes,
                static_cast<int>(polar_updates.front().nl_w3.extent(0)));
            Kokkos::realloc(Kokkos::WithoutInitializing,
                polar_tp_out,num_nodes,
                polar_updates.front().readout_linear_1.input_dimension());
            const int ro_width=static_cast<int>(
                polar_updates.front().readout_linear_1.output_dimension());
            Kokkos::realloc(Kokkos::WithoutInitializing,
                polar_ro_l1_out,num_nodes,ro_width);
            Kokkos::realloc(Kokkos::WithoutInitializing,
                polar_ro_lmid_out,num_nodes,ro_width);
            const int gate_scalars=polar_updates.front().gate_scalar_count;
            const int gate_gates=polar_updates.front().gate_gate_count;
            const int gate_gated=polar_updates.front().gate_gated_count;
            const int gate_out_width=gate_scalars+3*gate_gated;
            Kokkos::realloc(Kokkos::WithoutInitializing,
                polar_gate1_out,num_nodes,gate_out_width);
            Kokkos::realloc(Kokkos::WithoutInitializing,
                polar_gate2_out,num_nodes,gate_out_width);
            Kokkos::realloc(Kokkos::WithoutInitializing,
                polar_update_out,num_nodes,
                polar_updates.front().readout_linear_2.output_dimension());
            // Plan B1: allocate the per-step reverse cache (MATOS_REPLAY_CACHE).
            const bool replay_cache_on=polar_replay_cache_enabled();
            if(replay_cache_on) {
                const int pe_width=static_cast<int>(
                    polar_updates.front().potential_linear
                        .output_dimension());
                const int nl_hidden=
                    polar_updates.front().nl_hidden_width;
                const int nl_out_width=static_cast<int>(
                    polar_updates.front().nl_w3.extent(0));
                const int out_width=static_cast<int>(
                    polar_updates.front().readout_linear_2
                        .output_dimension());
                ensure_view(polar_rev_cache_pe,polar_rev_cache_pe_storage,
                    polar_recursion_steps,num_nodes,pe_width);
                ensure_view(polar_rev_cache_nl_h,
                    polar_rev_cache_nl_h_storage,polar_recursion_steps,
                    num_nodes,3*nl_hidden);
                ensure_view(polar_rev_cache_nl_a,
                    polar_rev_cache_nl_a_storage,polar_recursion_steps,
                    num_nodes,3*nl_hidden);
                ensure_view(polar_rev_cache_nl_out,
                    polar_rev_cache_nl_out_storage,polar_recursion_steps,
                    num_nodes,nl_out_width);
                ensure_view(polar_rev_cache_ro1,
                    polar_rev_cache_ro1_storage,polar_recursion_steps,
                    num_nodes,ro_width);
                ensure_view(polar_rev_cache_ro2,
                    polar_rev_cache_ro2_storage,polar_recursion_steps,
                    num_nodes,ro_width);
                ensure_view(polar_rev_cache_out,
                    polar_rev_cache_out_storage,polar_recursion_steps,
                    num_nodes,out_width);
                polar_rev_cache_valid=false;
            }

            auto kcos2=polar_cosines;
            auto ksin2=polar_sines;
            auto dbre=polar_density_basis_re;
            auto dbim=polar_density_basis_im;
            auto fbre=polar_feature_basis_re;
            auto fbim=polar_feature_basis_im;
            auto kn2=polar_k_norm2;
            auto k0m=polar_k0_mask;
            auto density_all=polar_working_density;
            auto feats_a=polar_field_feats_a;
            auto feats_b=polar_field_feats_b;
            auto step_potential=polar_step_potential_features;
            auto knorm=polar_field_feature_norms;
            auto permutation_view=polar_output_permutation;
            auto si_overlap_view=polar_si_overlap;
            auto si_select_view=polar_si_select;
            auto kweight=polar_k_weight;
            auto features_mixed_view=polar_features_mixed;
            auto pe_view=polar_pe_out;
            auto scratch_a=polar_scratch_a;
            auto scratch_b=polar_scratch_b;
            auto dp_view=polar_dp_out;
            auto se_view=polar_se_out;
            auto nl_in_view=polar_nl_input;
            auto nl_a_view=polar_nl_a;
            auto nl_b_view=polar_nl_b;
            auto nl_out_view=polar_nl_out;
            auto tp_view=polar_tp_out;
            auto ro1_view=polar_ro_l1_out;
            auto gate1_view=polar_gate1_out;
            auto ro2_view=polar_ro_lmid_out;
            auto gate2_view=polar_gate2_out;
            auto update_out_view=polar_update_out;
            auto step_densities=polar_scf_step_densities;
            auto attrs_view=attrs;
            for(int step=0;step<polar_recursion_steps;++step) {
                auto& update=polar_updates[step];
                // 路径表主机副本(每步取一次;极小;std::vector 传参规避
                // nvcc "泛型 lambda 内不允许 device lambda" 限制)
                const auto to_host_vector=[&](const auto& view) {
                    const auto mirror=Kokkos::create_mirror_view_and_copy(
                        Kokkos::HostSpace(),view);
                    std::vector<typename decltype(mirror)::value_type>
                        values(mirror.size());
                    for(std::size_t i2=0;i2<values.size();++i2)
                        values[i2]=mirror.data()[i2];
                    return values;
                };
                const std::vector<int> dot_meta=
                    to_host_vector(update.uvu_dot_meta);
                const std::vector<Precision> dot_path_weight=
                    to_host_vector(update.uvu_dot_path_weight);
                const std::vector<int> tp_meta=
                    to_host_vector(update.uvu_tp_meta);
                const std::vector<Precision> tp_path_weight=
                    to_host_vector(update.uvu_tp_path_weight);
                auto dot_weight=update.uvu_dot_weight;
                auto dot_mask=update.uvu_dot_mask;
                auto tp_weight=update.uvu_tp_weight;
                auto tp_mask=update.uvu_tp_mask;
                const auto launch_uvu=[&](const std::vector<int>& meta,
                        const std::vector<Precision>& path_weight,
                        Kokkos::View<Precision*> weight_view,
                        Kokkos::View<Precision*> mask_view,
                        Kokkos::View<Precision**,Kokkos::LayoutRight> x1,
                        Kokkos::View<Precision**,Kokkos::LayoutRight> x2,
                        Kokkos::View<Precision**,Kokkos::LayoutRight> output) {
                    const int paths=static_cast<int>(meta.size())/8;
                    Kokkos::parallel_for("polar uvu zero",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                            {num_nodes,
                             static_cast<int>(output.extent(1))}),
                        KOKKOS_LAMBDA(int node,int column) {
                            output(node,column)=Precision(0);
                        });
                    for(int p=0;p<paths;++p) {
                        const int in1_off=meta[p*8+0],in2_off=meta[p*8+1],
                            out_off=meta[p*8+2],w_off=meta[p*8+3],
                            mul1=meta[p*8+4],mul2=meta[p*8+5],
                            mode=meta[p*8+6],d1=meta[p*8+7];
                        const Precision pw=path_weight[p];
                        const Precision inv_sqrt_d1=Precision(1)
                            /Kokkos::sqrt(Precision(d1));
                        if(mode==0) {
                            if(d1<=8) {
                                const char* sweep_env=std::getenv(
                                    "MATOS_UVU_FWD_SWEEP");
                                int sweep=4;
                                if(sweep_env!=nullptr)
                                    sweep=std::atoi(sweep_env);
                                if(sweep!=2&&sweep!=4&&sweep!=8) sweep=4;
                                int tile_n=32,tile_m=32;
                                uvu_tile_shape(tile_n,tile_m);
                                const int n_tiles=
                                    (num_nodes+tile_n-1)/tile_n;
                                const int m_tiles=
                                    (mul1+tile_m-1)/tile_m;
                                using scratch_view=Kokkos::View<Precision*,
                                    Kokkos::DefaultExecutionSpace::
                                        scratch_memory_space,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
                                if(sweep>1) {
                                    // Round-3 forward UVU split: S teams per
                                    // (node,u) tile, each owns a contiguous v
                                    // chunk; a second kernel folds the
                                    // partials in ascending chunk order.
                                    const int chunks=sweep;
                                    const int chunk_width=
                                        (mul2+chunks-1)/chunks;
                                    const int stride=chunk_width+1;
                                    const std::size_t smem=
                                        scratch_view::shmem_size(
                                            tile_m*stride);
                                    ensure_view(polar_fwd_uvu_partial,
                                        polar_fwd_uvu_partial_storage,
                                        chunks,num_nodes,mul1);
                                    auto partial=polar_fwd_uvu_partial;
                                    Kokkos::parallel_for(
                                        "polar uvu mode0",
                                        Kokkos::TeamPolicy<>(
                                            n_tiles*m_tiles*chunks,
                                            tile_n*tile_m)
                                            .set_scratch_size(0,
                                                Kokkos::PerTeam(smem)),
                                        KOKKOS_LAMBDA(
                                            const Kokkos::TeamPolicy<>::
                                                member_type& team) {
                                            const int tile=
                                                team.league_rank();
                                            const int chunk_index=
                                                tile%chunks;
                                            const int block=tile/chunks;
                                            const int nt=block/m_tiles,
                                                ut=block%m_tiles;
                                            const int n0=nt*tile_n,
                                                u0=ut*tile_m;
                                            const int v_begin=
                                                chunk_index*chunk_width;
                                            const int v_end=Kokkos::min(
                                                mul2,v_begin+chunk_width);
                                            const int width=v_end-v_begin;
                                            scratch_view s(
                                                team.team_scratch(0),
                                                tile_m*stride);
                                            for(int idx=team.team_rank();
                                                idx<tile_m*width;
                                                idx+=team.team_size()) {
                                                const int uu=idx/width,
                                                    vv=idx%width;
                                                const int u=u0+uu;
                                                s(uu*stride+vv)=(u<mul1)
                                                    ?weight_view(w_off
                                                        +u*mul2+v_begin+vv)
                                                    :Precision(0);
                                            }
                                            team.team_barrier();
                                            const int tr=team.team_rank();
                                            const int node=n0+tr/tile_m,
                                                uu=tr%tile_m;
                                            if(node<num_nodes
                                                &&u0+uu<mul1) {
                                                const int u=u0+uu;
                                                Precision x1v[8];
                                                for(int c=0;c<d1;++c)
                                                    x1v[c]=x1(node,
                                                        in1_off+u*d1+c);
                                                Precision mixed=
                                                    Precision(0);
                                                for(int v=v_begin;v<v_end;
                                                    ++v) {
                                                    Precision pair=
                                                        Precision(0);
                                                    for(int c=0;c<d1;++c)
                                                        pair+=x1v[c]
                                                            *x2(node,
                                                                in2_off
                                                                    +v*d1+c);
                                                    mixed+=pair
                                                        *inv_sqrt_d1
                                                        *s(uu*stride
                                                            +v-v_begin);
                                                }
                                                partial(chunk_index,
                                                    node,u)=mixed;
                                            }
                                        });
                                    Kokkos::parallel_for(
                                        "polar uvu mode0 reduce",
                                        Kokkos::MDRangePolicy<
                                            Kokkos::Rank<2>>(
                                            {0,0},{num_nodes,mul1}),
                                        KOKKOS_LAMBDA(int node,int u) {
                                            Precision total=Precision(0);
                                            for(int chunk_index=0;
                                                chunk_index<chunks;
                                                ++chunk_index)
                                                total+=partial(chunk_index,
                                                    node,u);
                                            output(node,out_off+u)+=pw*total;
                                        });
                                } else {
                                // Single-pass (sweep=1): stage the weight
                                // tile in ascending v chunks so the shared
                                // memory footprint stays bounded while the
                                // accumulation order is unchanged.
                                const int stage_v=64;
                                const int stride2=stage_v+1;
                                const std::size_t smem=
                                    scratch_view::shmem_size(
                                        tile_m*stride2);
                                Kokkos::parallel_for("polar uvu mode0",
                                    Kokkos::TeamPolicy<>(
                                        n_tiles*m_tiles,tile_n*tile_m)
                                        .set_scratch_size(0,
                                            Kokkos::PerTeam(smem)),
                                    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::
                                        member_type& team) {
                                        const int tile=team.league_rank();
                                        const int nt=tile/m_tiles,
                                            ut=tile%m_tiles;
                                        const int n0=nt*tile_n,
                                            u0=ut*tile_m;
                                        scratch_view s(
                                            team.team_scratch(0),
                                            tile_m*stride2);
                                        const int tr=team.team_rank();
                                        const int node=n0+tr/tile_m,
                                            uu=tr%tile_m;
                                        const bool active=node<num_nodes
                                            &&u0+uu<mul1;
                                        const int u=u0+uu;
                                        Precision x1v[8];
                                        if(active)
                                            for(int c=0;c<d1;++c)
                                                x1v[c]=x1(node,
                                                    in1_off+u*d1+c);
                                        Precision mixed=Precision(0);
                                        for(int vbegin=0;vbegin<mul2;
                                            vbegin+=stage_v) {
                                            const int width=Kokkos::min(
                                                stage_v,mul2-vbegin);
                                            for(int idx=tr;
                                                idx<tile_m*width;
                                                idx+=team.team_size()) {
                                                const int uu2=idx/width,
                                                    vv=idx%width;
                                                const int u2=u0+uu2;
                                                s(uu2*stride2+vv)=
                                                    (u2<mul1)
                                                    ?weight_view(w_off
                                                        +u2*mul2
                                                        +vbegin+vv)
                                                    :Precision(0);
                                            }
                                            team.team_barrier();
                                            if(active) {
                                                for(int v=0;v<width;++v) {
                                                    Precision pair=
                                                        Precision(0);
                                                    for(int c=0;c<d1;++c)
                                                        pair+=x1v[c]
                                                            *x2(node,
                                                                in2_off
                                                                    +(vbegin+v)
                                                                        *d1+c);
                                                    mixed+=pair
                                                        *inv_sqrt_d1
                                                        *s(uu*stride2+v);
                                                }
                                            }
                                            team.team_barrier();
                                        }
                                        if(active)
                                            output(node,out_off+u)
                                                +=pw*mixed;
                                    });
                                }
                            } else {
                            Kokkos::parallel_for("polar uvu mode0",
                                Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                                    {0,0},{num_nodes,mul1}),
                                KOKKOS_LAMBDA(int node,int u) {
                                    Precision mixed=Precision(0);
                                    for(int v=0;v<mul2;++v) {
                                        Precision pair=Precision(0);
                                        for(int c2=0;c2<d1;++c2)
                                            pair+=x1(node,
                                                    in1_off+u*d1+c2)
                                                *x2(node,
                                                    in2_off+v*d1+c2);
                                        mixed+=pair*inv_sqrt_d1
                                            *weight_view(
                                                w_off+u*mul2+v);
                                    }
                                    output(node,out_off+u)+=pw*mixed;
                                });
                            }
                        } else {
                            Kokkos::parallel_for("polar uvu mode1",
                                Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                                    {0,0},{num_nodes,mul1}),
                                KOKKOS_LAMBDA(int node,int u) {
                                    Precision mixed=Precision(0);
                                    for(int v=0;v<mul2;++v)
                                        mixed+=x2(node,in2_off+v)
                                            *weight_view(
                                                w_off+u*mul2+v);
                                    const Precision contribution=
                                        pw*mixed*inv_sqrt_d1;
                                    for(int c2=0;c2<d1;++c2)
                                        output(node,
                                                out_off+u*d1+c2)+=
                                            contribution*x1(node,
                                                in1_off+u*d1+c2);
                                });
                        }
                    }
                    Kokkos::parallel_for("polar uvu mask",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                            {num_nodes,
                             static_cast<int>(output.extent(1))}),
                        KOKKOS_LAMBDA(int node,int column) {
                            output(node,column)*=mask_view(column);
                        });
                };
                for(int channel=0;channel<2;++channel) {
                    auto field_out=channel==0?feats_a:feats_b;
                    Kokkos::parallel_for("polar scf assemble fourier",num_k,
                        KOKKOS_LAMBDA(int k) {
                            Precision acc_cos[4],acc_sin[4];
                            for(int m=0;m<4;++m) {
                                acc_cos[m]=Precision(0);
                                acc_sin[m]=Precision(0);
                            }
                            for(int node=0;node<owned_nodes;++node) {
                                const Precision c=kcos2(k,node);
                                const Precision s=ksin2(k,node);
                                for(int m=0;m<4;++m) {
                                    const Precision q=density_all(
                                        node,4*channel+m);
                                    acc_cos[m]+=c*q;
                                    acc_sin[m]+=s*q;
                                }
                            }
                            Precision re=Precision(0),im=Precision(0);
                            for(int m=0;m<4;++m) {
                                re+=dbre(k,m)*acc_cos[m]
                                    +dbim(k,m)*acc_sin[m];
                                im+=dbim(k,m)*acc_cos[m]
                                    -dbre(k,m)*acc_sin[m];
                            }
                            step_rho_re(k)=scf_rho_scale*re;
                            step_rho_im(k)=scf_rho_scale*im;
                        });
                    polar_global_sum_view(step_rho_re);
                    polar_global_sum_view(step_rho_im);
                    Kokkos::parallel_for("polar scf project features",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                            {num_nodes,8}),
                        KOKKOS_LAMBDA(int node,int j) {
                            const int m=permutation_view(j);
                            Precision proj_cos=Precision(0);
                            Precision proj_sin=Precision(0);
                            for(int k=0;k<num_k;++k) {
                                const Precision inv_k2=(k0m(k)
                                        >Precision(0))
                                    ?Precision(0)
                                    :Precision(scf_field_constant)
                                        /kn2(k);
                                const Precision pr=step_rho_re(k)
                                    *inv_k2;
                                const Precision pi=step_rho_im(k)
                                    *inv_k2;
                                const Precision kfp=(k0m(k)
                                        >Precision(0))
                                    ?Precision(0.5):Precision(1);
                                const Precision fa=(pr*fbre(k,m)
                                    +pi*fbim(k,m))*kfp;
                                const Precision fb=(pr*fbim(k,m)
                                    -pi*fbre(k,m))*kfp;
                                proj_cos+=(kweight(k)*fa)*kcos2(k,node);
                                proj_sin+=(kweight(k)*fb)*ksin2(k,node);
                            }
                            field_out(node,j)=scf_proj_scale
                                *(proj_cos+proj_sin)
                                -si_overlap_view(j)*density_all(
                                    node,4*channel+si_select_view(j));
                        });
                }
                Kokkos::parallel_for("polar scf normalize potential features",
                    num_nodes,KOKKOS_LAMBDA(int node) {
                        for(int j=0;j<8;++j) {
                            step_potential(node,j)=feats_a(node,j)/knorm(j);
                            step_potential(node,8+j)=
                                feats_b(node,j)/knorm(j);
                        }
                    });
                // 更新块前向(field_blocks.py L563-594)
                update.potential_linear.evaluate(step_potential,pe_view);
                update.node_feats_linear.evaluate(
                    features_mixed_view,scratch_a);
                update.charge_embedding.evaluate(density_all,scratch_b);
                Kokkos::parallel_for("polar pe sum",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                        {num_nodes,static_cast<int>(pe_view.extent(1))}),
                    KOKKOS_LAMBDA(int node,int column) {
                        pe_view(node,column)+=scratch_a(node,column)
                            +scratch_b(node,column);
                    });
                launch_uvu(dot_meta,dot_path_weight,dot_weight,dot_mask,
                    features_mixed_view,pe_view,dp_view);
                update.source_embedding.evaluate(attrs_view,se_view);
                Kokkos::parallel_for("polar nl concat",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                        {num_nodes,dp_width}),
                    KOKKOS_LAMBDA(int node,int column) {
                        nl_in_view(node,column)=dp_view(node,column);
                        nl_in_view(node,dp_width+column)=
                            se_view(node,column);
                    });
                // MLP: [linear, LN, silu]×3 + linear(radial FullyConnectedNet)
                {
                    auto w0=update.nl_w0;
                    auto b0=update.nl_b0;
                    Kokkos::parallel_for("polar nl dense0",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                            {num_nodes,
                             static_cast<int>(w0.extent(0))}),
                        KOKKOS_LAMBDA(int node,int o) {
                            Precision acc=b0(o);
                            for(int c2=0;c2<static_cast<int>(w0.extent(1));
                                ++c2)
                                acc+=w0(o,c2)*nl_in_view(node,c2);
                            nl_a_view(node,o)=acc;
                        });
                }
                if(replay_cache_on)
                    Kokkos::deep_copy(Kokkos::subview(
                            polar_rev_cache_nl_h,step,Kokkos::ALL,
                            Kokkos::make_pair(0,update.nl_hidden_width)),
                        nl_a_view);
                const Precision ln_eps=update.ln_eps;
                const auto launch_ln_silu=[&](
                        Kokkos::View<Precision**,Kokkos::LayoutRight> src,
                        Kokkos::View<Precision**,Kokkos::LayoutRight> dst,
                        Kokkos::View<Precision*> gamma,
                        Kokkos::View<Precision*> beta) {
                    const int width=static_cast<int>(src.extent(1));
                    const int ln_team=width<128?width:128;
                    Kokkos::parallel_for("polar layernorm silu",
                        Kokkos::TeamPolicy<>(num_nodes,ln_team),
                        KOKKOS_LAMBDA(
                            const Kokkos::TeamPolicy<>::member_type& team) {
                            const int node=team.league_rank();
                            Precision mean=Precision(0);
                            Kokkos::parallel_reduce(
                                Kokkos::TeamThreadRange(team,width),
                                [&](const int c2,Precision& sum) {
                                    sum+=src(node,c2);
                                },mean);
                            mean/=Precision(width);
                            Precision var=Precision(0);
                            Kokkos::parallel_reduce(
                                Kokkos::TeamThreadRange(team,width),
                                [&](const int c2,Precision& sum) {
                                    const Precision delta=
                                        src(node,c2)-mean;
                                    sum+=delta*delta;
                                },var);
                            var/=Precision(width);
                            const Precision inv=Precision(1)
                                /Kokkos::sqrt(var+ln_eps);
                            Kokkos::parallel_for(
                                Kokkos::TeamThreadRange(team,width),
                                [&](const int c2) {
                                    const Precision normalized=(src(node,c2)
                                        -mean)*inv*gamma(c2)+beta(c2);
                                    dst(node,c2)=normalized
                                        /(Precision(1)+Kokkos::exp(
                                            -normalized));
                                });
                        });
                };
                launch_ln_silu(nl_a_view,nl_b_view,
                    update.ln0_weight,update.ln0_bias);
                if(replay_cache_on)
                    Kokkos::deep_copy(Kokkos::subview(
                            polar_rev_cache_nl_a,step,Kokkos::ALL,
                            Kokkos::make_pair(0,update.nl_hidden_width)),
                        nl_b_view);
                {
                    auto w1=update.nl_w1;
                    auto b1=update.nl_b1;
                    Kokkos::parallel_for("polar nl dense1",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                            {num_nodes,
                             static_cast<int>(w1.extent(0))}),
                        KOKKOS_LAMBDA(int node,int o) {
                            Precision acc=b1(o);
                            for(int c2=0;c2<static_cast<int>(w1.extent(1));
                                ++c2)
                                acc+=w1(o,c2)*nl_b_view(node,c2);
                            nl_a_view(node,o)=acc;
                        });
                }
                if(replay_cache_on)
                    Kokkos::deep_copy(Kokkos::subview(
                            polar_rev_cache_nl_h,step,Kokkos::ALL,
                            Kokkos::make_pair(update.nl_hidden_width,
                                2*update.nl_hidden_width)),
                        nl_a_view);
                launch_ln_silu(nl_a_view,nl_b_view,
                    update.ln1_weight,update.ln1_bias);
                if(replay_cache_on)
                    Kokkos::deep_copy(Kokkos::subview(
                            polar_rev_cache_nl_a,step,Kokkos::ALL,
                            Kokkos::make_pair(update.nl_hidden_width,
                                2*update.nl_hidden_width)),
                        nl_b_view);
                {
                    auto w2=update.nl_w2;
                    auto b2=update.nl_b2;
                    Kokkos::parallel_for("polar nl dense2",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                            {num_nodes,
                             static_cast<int>(w2.extent(0))}),
                        KOKKOS_LAMBDA(int node,int o) {
                            Precision acc=b2(o);
                            for(int c2=0;c2<static_cast<int>(w2.extent(1));
                                ++c2)
                                acc+=w2(o,c2)*nl_b_view(node,c2);
                            nl_a_view(node,o)=acc;
                        });
                }
                if(replay_cache_on)
                    Kokkos::deep_copy(Kokkos::subview(
                            polar_rev_cache_nl_h,step,Kokkos::ALL,
                            Kokkos::make_pair(2*update.nl_hidden_width,
                                3*update.nl_hidden_width)),
                        nl_a_view);
                launch_ln_silu(nl_a_view,nl_b_view,
                    update.ln2_weight,update.ln2_bias);
                if(replay_cache_on)
                    Kokkos::deep_copy(Kokkos::subview(
                            polar_rev_cache_nl_a,step,Kokkos::ALL,
                            Kokkos::make_pair(2*update.nl_hidden_width,
                                3*update.nl_hidden_width)),
                        nl_b_view);
                {
                    auto w3=update.nl_w3;
                    auto b3=update.nl_b3;
                    Kokkos::parallel_for("polar nl dense3",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                            {num_nodes,
                             static_cast<int>(w3.extent(0))}),
                        KOKKOS_LAMBDA(int node,int o) {
                            Precision acc=b3(o);
                            for(int c2=0;c2<static_cast<int>(w3.extent(1));
                                ++c2)
                                acc+=w3(o,c2)*nl_b_view(node,c2);
                            nl_out_view(node,o)=acc;
                        });
                }
                launch_uvu(tp_meta,tp_path_weight,tp_weight,tp_mask,
                    features_mixed_view,nl_out_view,tp_view);
                // 读出: linear_1 → Gate → linear_mid → Gate → linear_2
                update.readout_linear_1.evaluate(tp_view,ro1_view);
                const Precision gate_scalar_cst=update.gate_scalar_cst;
                const Precision gate_gate_cst=update.gate_gate_cst;
                const int gate_scalars=update.gate_scalar_count;
                const int gate_gates=update.gate_gate_count;
                const int gate_gated=update.gate_gated_count;
                const auto launch_readout_gate=[&](
                        Kokkos::View<Precision**,Kokkos::LayoutRight> src,
                        Kokkos::View<Precision**,Kokkos::LayoutRight> dst) {
                    Kokkos::parallel_for("polar readout gate",num_nodes,
                        KOKKOS_LAMBDA(int node) {
                            for(int i2=0;i2<gate_scalars;++i2) {
                                const Precision s=src(node,i2);
                                dst(node,i2)=gate_scalar_cst*s
                                    /(Precision(1)+Kokkos::exp(-s));
                            }
                            for(int i2=0;i2<gate_gates;++i2) {
                                const Precision g=gate_gate_cst
                                    /(Precision(1)+Kokkos::exp(
                                        -src(node,gate_scalars+i2)));
                                for(int c2=0;c2<3;++c2)
                                    dst(node,gate_scalars+3*i2+c2)=g
                                        *src(node,gate_scalars+gate_gates
                                            +3*i2+c2);
                            }
                        });
                    (void)gate_gated;
                };
                launch_readout_gate(ro1_view,gate1_view);
                update.readout_linear_mid.evaluate(gate1_view,ro2_view);
                launch_readout_gate(ro2_view,gate2_view);
                update.readout_linear_2.evaluate(gate2_view,update_out_view);
                // 密度更新(前 8 列;extensions.py L836-839)
                Kokkos::parallel_for("polar density update",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                        {num_nodes,8}),
                    KOKKOS_LAMBDA(int node,int column) {
                        density_all(node,column)+=
                            update_out_view(node,column);
                    });
                // 逐步 Fukui 重归一+l=0 校正(extensions.py L841-866)
                const auto sources_host=
                    Kokkos::create_mirror_view_and_copy(
                        Kokkos::HostSpace(),update_out_view);
                const auto density_host=
                    Kokkos::create_mirror_view_and_copy(
                        Kokkos::HostSpace(),density_all);
                Precision source_sums[2]={Precision(0),Precision(0)};
                Precision predicted_zero[2]={Precision(0),Precision(0)};
                for(int node=0;node<owned_nodes;++node) {
                    source_sums[0]+=sources_host(node,8);
                    source_sums[1]+=sources_host(node,9);
                    predicted_zero[0]+=density_host(node,0);
                    predicted_zero[1]+=density_host(node,4);
                }
                {
                    double global_sums[4]={static_cast<double>(source_sums[0]),
                        static_cast<double>(source_sums[1]),
                        static_cast<double>(predicted_zero[0]),
                        static_cast<double>(predicted_zero[1])};
                    polar_global_sum_host(global_sums,4);
                    source_sums[0]=Precision(global_sums[0]);
                    source_sums[1]=Precision(global_sums[1]);
                    predicted_zero[0]=Precision(global_sums[2]);
                    predicted_zero[1]=Precision(global_sums[3]);
                }
                Precision raw_source_sums[2]={source_sums[0],source_sums[1]};
                for(int channel=0;channel<2;++channel)
                    if(source_sums[channel]==Precision(0))
                        source_sums[channel]=Precision(1);
                const Precision charge=Precision(total_charge);
                const Precision spin_multiplicity=Precision(total_spin);
                const Precision target_alpha=Precision(0.5)
                    *(charge+(spin_multiplicity-Precision(1)));
                const Precision target_beta=Precision(0.5)
                    *(charge-(spin_multiplicity-Precision(1)));
                const Precision factor_alpha=(target_alpha
                    -predicted_zero[0])/source_sums[0];
                const Precision factor_beta=(target_beta
                    -predicted_zero[1])/source_sums[1];
                Kokkos::parallel_for("polar scf fukui correction",num_nodes,
                    KOKKOS_LAMBDA(int node) {
                        density_all(node,0)+=update_out_view(node,8)
                            *factor_alpha;
                        density_all(node,4)+=update_out_view(node,9)
                            *factor_beta;
                    });
                // P3 模块⑧: 逐步 Fukui 标量缓存([Sα,Sβ,predα,predβ,Fα,Fβ]),供反向一次性消费。
                // [v3 适配·D-03] 校后密度链直接消费既有 polar_scf_step_densities——其
                // "polar scf density snapshot" 内核位于本 Fukui 校正之后(校后口径, 与提案
                // step_densities_post 同值同源); 原提案的重复快照视图/内核不再新增。
                if(static_cast<int>(polar_scf_fukui_scalars.size())
                    !=polar_recursion_steps)
                    polar_scf_fukui_scalars.assign(polar_recursion_steps,
                        std::array<Precision,6>{});
                polar_scf_fukui_scalars[step]={
                    raw_source_sums[0],raw_source_sums[1],
                    predicted_zero[0],predicted_zero[1],
                    factor_alpha,factor_beta};
                // 快照 = 逐步 Fukui 校正后的完整状态(参考口径 scf_steps step{i+1})
                Kokkos::parallel_for("polar scf density snapshot",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                        {num_nodes,8}),
                    KOKKOS_LAMBDA(int node,int column) {
                        step_densities(step,node,column)=
                            density_all(node,column);
                    });
                // Plan B1: values used by the reverse; they are not
                // overwritten after this point in the step.
                if(replay_cache_on) {
                    Kokkos::deep_copy(Kokkos::subview(
                            polar_rev_cache_pe,step,Kokkos::ALL,
                            Kokkos::ALL),pe_view);
                    Kokkos::deep_copy(Kokkos::subview(
                            polar_rev_cache_nl_out,step,Kokkos::ALL,
                            Kokkos::ALL),nl_out_view);
                    Kokkos::deep_copy(Kokkos::subview(
                            polar_rev_cache_ro1,step,Kokkos::ALL,
                            Kokkos::ALL),ro1_view);
                    Kokkos::deep_copy(Kokkos::subview(
                            polar_rev_cache_ro2,step,Kokkos::ALL,
                            Kokkos::ALL),ro2_view);
                    Kokkos::deep_copy(Kokkos::subview(
                            polar_rev_cache_out,step,Kokkos::ALL,
                            Kokkos::ALL),update_out_view);
                }
            }
            if(replay_cache_on) polar_rev_cache_valid=true;
            Kokkos::realloc(Kokkos::WithoutInitializing,
                polar_scf_final_density,num_nodes,8);
            auto final_density=polar_scf_final_density;
            Kokkos::parallel_for("polar scf final density",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                    {num_nodes,8}),
                KOKKOS_LAMBDA(int node,int column) {
                    final_density(node,column)=density_all(node,column);
                });
        }
        // [v2 适配] ⑥⑦ 运行期合计段: ⑥-C1(module6 §3.2 原文, 8 空格缩进不变)
        // + ⑦-4.3/4.4(module7 原文, 整体 +4 缩进; 4.3 末态密度代名 spin_charge
        //   _density -> polar_scf_final_density, 见行内注释) + ⑥-C2(module6 §3.3
        // 原文, +4 缩进; energies 局部别名 -> 成员 node_energies)。插入点=⑤ 块之后、函数尾前。
        symmetrix_polar_profile_mark("stage6:coulomb");
        // ---- P3 模块⑥: coulomb_energy(全 k GTO-Ewald 基线) ----
        // 依据: graph_longrange v0.4.0 energy.py L113-166(_pbc_energy_batch)与
        // features.py L28-83(结构因子/库仑核), gto_utils.py L283-302(自相互作用);
        // 密度源 = SCF 末态(extensions.py L887-893 自旋求和, L910-924 调用)。
        // 公式经 bench/polar_module6_preflight.py 对 4 夹具双源逐位锁定(rel ≤1.2e-15)。
        // 化简: E = FIELD_CONSTANT/(2V)·Σ_k 2·kfac·(ρ̂_r²+ρ̂_i²), kfac=1/k²(k=0 跳过;
        // 由 density=(2π)³ρ̂/V 对 energy.py L17-30 两次代入所得,与预检链等价)。
        // 作用域: num_k/volume 为③ 块局部量,本块与③④⑤ 同处一个
        // if(polar_enabled){...} 内 ⇒ 直接可见(不得移到块外,见 §3.1)。
        {
            constexpr double coulomb_field_constant=1.0/(5.526349406e-3);
            // (a) ρ̂(k) = Σ_n e^{ik·r_n}·B(k)·q_n(未乘 (2π)³/V;q_n = 自旋和)
            ensure_view(polar_coulomb_rho_re,polar_coulomb_rho_re_storage,
                std::size_t(num_k));
            ensure_view(polar_coulomb_rho_im,polar_coulomb_rho_im_storage,
                std::size_t(num_k));
            auto coulomb_rho_re=polar_coulomb_rho_re;
            auto coulomb_rho_im=polar_coulomb_rho_im;
            auto coulomb_cos=polar_cosines;
            auto coulomb_sin=polar_sines;
            auto coulomb_dbre=polar_density_basis_re;
            auto coulomb_dbim=polar_density_basis_im;
            // 密度源 = ⑤ 写入的 SCF 末态快照(N,8)。若在⑤ 未落地的分支打本补丁,
            // 把下一行换成 polar_field_independent_density(② 出口;过渡门 §3.4)。
            auto coulomb_density=polar_scf_final_density;
            Kokkos::parallel_for("polar coulomb rho",num_k,KOKKOS_LAMBDA(int k) {
                Precision acc_cos[4]={Precision(0),Precision(0),Precision(0),
                    Precision(0)};
                Precision acc_sin[4]={Precision(0),Precision(0),Precision(0),
                    Precision(0)};
                for(int node=0;node<owned_nodes;++node) {
                    const Precision c=coulomb_cos(k,node);
                    const Precision s=coulomb_sin(k,node);
                    for(int m=0;m<4;++m) {
                        const Precision q=coulomb_density(node,m)
                            +coulomb_density(node,4+m);
                        acc_cos[m]+=c*q;
                        acc_sin[m]+=s*q;
                    }
                }
                Precision re=Precision(0),im=Precision(0);
                for(int m=0;m<4;++m) {
                    re+=coulomb_dbre(k,m)*acc_cos[m]
                        +coulomb_dbim(k,m)*acc_sin[m];
                    im+=coulomb_dbim(k,m)*acc_cos[m]
                        -coulomb_dbre(k,m)*acc_sin[m];
                }
                coulomb_rho_re(k)=re;
                coulomb_rho_im(k)=im;
            });
            polar_global_sum_view(coulomb_rho_re);
            polar_global_sum_view(coulomb_rho_im);
            // (b) E_k 归约(k=0 跳过;标量用 double,与库内既有 parallel_reduce
            //     标量风格一致:tpp "validate nonlinear nodes"/"reduce_stress")
            auto coulomb_kn2=polar_k_norm2;
            auto coulomb_k0=polar_k0_mask;
            auto coulomb_kweight=polar_k_weight;   // P4 RBE: 确定性=1
            double coulomb_k_energy=0.0;
            Kokkos::parallel_reduce("polar coulomb k energy",num_k,
                KOKKOS_LAMBDA(int k,double& acc) {
                    if(coulomb_k0(k)>Precision(0)) return;
                    const double re=static_cast<double>(coulomb_rho_re(k));
                    const double im=static_cast<double>(coulomb_rho_im(k));
                    const double w=static_cast<double>(coulomb_kweight(k));
                    acc+=w*2.0*(re*re+im*im)
                        /static_cast<double>(coulomb_kn2(k));
                },coulomb_k_energy);
            double coulomb_energy_value=coulomb_k_energy
                *coulomb_field_constant/(2.0*volume);
            // (c) 自相互作用(GTO self block,energy.py L156-166 减 0.5·Σ<q,self>)。
            //     本模型序列化 include_self_interaction=true ⇒ 分支不执行
            //     (反事实诊断在门脚本)。表 = coulomb 自己的 [4] 表。
            if(!polar_include_self_interaction) {
                auto coulomb_si_overlap=polar_coulomb_si_overlap;
                auto coulomb_si_select=polar_coulomb_si_select;
                double coulomb_self_energy=0.0;
                Kokkos::parallel_reduce("polar coulomb self interaction",
                    num_nodes,KOKKOS_LAMBDA(int node,double& acc) {
                        double value=0.0;
                        for(int m=0;m<4;++m) {
                            const int selected=coulomb_si_select(m);
                            const double q=static_cast<double>(
                                coulomb_density(node,m)
                                +coulomb_density(node,4+m));
                            const double q_selected=static_cast<double>(
                                coulomb_density(node,selected)
                                +coulomb_density(node,4+selected));
                            value+=q*static_cast<double>(
                                coulomb_si_overlap(m))*q_selected;
                        }
                        acc+=value;
                    },coulomb_self_energy);
                coulomb_energy_value-=0.5*coulomb_self_energy;
            }
            // (d) pbc 修正(energy.py L168-192): 本路径 = 全周期(TTT)盒,
            //     is_molecule/is_slab 选择器均为假(slabs.py L185-256/L59-69 两项
            //     不执行)⇒ 修正恒 0。include_pbc_corrections=true 的语义即
            //     "该恒 0 项已被计入"。非全周期盒暂不支持(引擎无 pbc 入参;
            //     后续 slice 显式实现 monopole/slab 解析式并拒绝未实现分支)。
            if(polar_include_pbc_corrections) {
                // TTT ⇒ 0(保留闸门语义与后续 slice 挂点)。
            }
            // (e) 标量落盘(device 端写;host 不得直接访存 CUDA 视图)
            ensure_view(polar_coulomb_energy,polar_coulomb_energy_storage,
                std::size_t(1));
            auto coulomb_energy=polar_coulomb_energy;
            const double coulomb_energy_host=coulomb_energy_value;
            Kokkos::parallel_for("polar coulomb energy store",1,
                KOKKOS_LAMBDA(int) {
                    coulomb_energy(0)=coulomb_energy_host;
                });
        }

        if(polar_enabled&&polar_le_enabled) {
            symmetrix_polar_profile_mark("stage7:le");
            // ---- P3 模块⑦: local_electron_energy（OneBodyMLPFieldReadout L652-752） ----
            // 输入口径 = extensions.py L867-880：q_in = charges_induced(⑤末态)
            // + charges_0(②校正后初态)；v_in = ⑤末次迭代势特征 α|β；node_feats =
            // ① 末层 e3nn 基张量。前向 = linear_up → 两条 uvu 内积 → concat → RadialMLP
            // → 逐原子 local_q_e；总量 = 图内和（单图=整盒，无缩放）。
            if(static_cast<int>(polar_step_potential_features.extent(0))!=num_nodes
                ||polar_step_potential_features.extent(1)!=16)
                throw std::runtime_error(
                    "polar_blocks: local_electron_energy requires module ⑤ SCF step "
                    "potential features before it runs.");
            const int le_dot_width=static_cast<int>(polar_le_q_mask.extent(0));
            const int le_up_width=polar_le_linear_up_q.output_dimension();
            ensure_view(polar_le_q_in,polar_le_q_in_storage,num_nodes,8);
            ensure_view(polar_le_q_up,polar_le_q_up_storage,num_nodes,le_up_width);
            ensure_view(polar_le_v_up,polar_le_v_up_storage,num_nodes,le_up_width);
            ensure_view(polar_le_dot_q,polar_le_dot_q_storage,num_nodes,le_dot_width);
            ensure_view(polar_le_dot_v,polar_le_dot_v_storage,num_nodes,le_dot_width);
            ensure_view(polar_le_invar,polar_le_invar_storage,num_nodes,
                2*le_dot_width);
            ensure_view(polar_le_mlp_out,polar_le_mlp_out_storage,num_nodes,1);
            ensure_view(polar_le_atom,polar_le_atom_storage,num_nodes);
            {
                auto q_in=polar_le_q_in;
                auto charges_zero=polar_field_independent_density;
                // [v2 适配] ⑤ 末态密度: 原提案取 polar_spin_charge_density, 该视图自
                // 树 2026-10-15 12:13(工作密度引入) 起为只读①原始累积; 改用同值快照成员。
                auto charges_final=polar_scf_final_density;
                Kokkos::parallel_for("polar le q_in",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{num_nodes,8}),
                    KOKKOS_LAMBDA(int node,int column) {
                        q_in(node,column)=charges_final(node,column)
                            +charges_zero(node,column);
                    });
            }
            polar_le_linear_up_q.evaluate(polar_le_q_in,polar_le_q_up);
            polar_le_linear_up_v.evaluate(
                polar_step_potential_features,polar_le_v_up);
            // 自写 uvu 内核：镜像 field_blocks.py L405-466 的 mode0(l×l→0)/mode1(l×0→l),
            // 逐路径乘序列化 path_weight（JSON 0.03125）；与⑤ launch_uvu 同构。
            {
                const auto launch_le_uvu=[&](const int nodes,
                        const std::vector<int>& meta_host,
                        const std::vector<Precision>& path_weight,
                        Kokkos::View<Precision*> weight_view,
                        Kokkos::View<Precision*> mask_view,
                        Kokkos::View<Precision**,Kokkos::LayoutRight> x1,
                        Kokkos::View<Precision**,Kokkos::LayoutRight> x2,
                        Kokkos::View<Precision**,Kokkos::LayoutRight> output) {
                    // 注: meta_host/path_weight 为 HostMirror, 形参必须 auto&（同⑤ launch_uvu）;
                    // 显式 Kokkos::View<Precision*> 无法绑定 HostSpace 镜像。
                    const int paths=static_cast<int>(meta_host.size())/8;
                    Kokkos::parallel_for("polar le uvu zero",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                            {nodes,static_cast<int>(output.extent(1))}),
                        KOKKOS_LAMBDA(int node,int column) {
                            output(node,column)=Precision(0);
                        });
                    for(int p=0;p<paths;++p) {
                        const int in1_off=meta_host[p*8+0],
                            in2_off=meta_host[p*8+1],
                            out_off=meta_host[p*8+2],
                            w_off=meta_host[p*8+3],
                            mul1=meta_host[p*8+4],mul2=meta_host[p*8+5],
                            mode=meta_host[p*8+6],d1=meta_host[p*8+7];
                        const Precision pw=path_weight[p];
                        const Precision inv_sqrt_d1=Precision(1)
                            /Kokkos::sqrt(Precision(d1));
                        if(mode==0) {
                            if(d1<=8) {
                                int tile_n=32,tile_m=32;
                                uvu_tile_shape(tile_n,tile_m);
                                const int n_tiles=
                                    (nodes+tile_n-1)/tile_n;
                                const int m_tiles=
                                    (mul1+tile_m-1)/tile_m;
                                using scratch_view=Kokkos::View<Precision*,
                                    Kokkos::DefaultExecutionSpace::
                                        scratch_memory_space,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
                                // Stage the weight tile in ascending v
                                // chunks: bounded smem, unchanged order.
                                const int stage_v=64;
                                const int stride2=stage_v+1;
                                const std::size_t smem=
                                    scratch_view::shmem_size(
                                        tile_m*stride2);
                                Kokkos::parallel_for("polar le uvu mode0",
                                    Kokkos::TeamPolicy<>(
                                        n_tiles*m_tiles,tile_n*tile_m)
                                        .set_scratch_size(0,
                                            Kokkos::PerTeam(smem)),
                                    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::
                                        member_type& team) {
                                        const int tile=team.league_rank();
                                        const int nt=tile/m_tiles,
                                            ut=tile%m_tiles;
                                        const int n0=nt*tile_n,
                                            u0=ut*tile_m;
                                        scratch_view s(
                                            team.team_scratch(0),
                                            tile_m*stride2);
                                        const int tr=team.team_rank();
                                        const int node=n0+tr/tile_m,
                                            uu=tr%tile_m;
                                        const bool active=node<nodes
                                            &&u0+uu<mul1;
                                        const int u=u0+uu;
                                        Precision x1v[8];
                                        if(active)
                                            for(int c=0;c<d1;++c)
                                                x1v[c]=x1(node,
                                                    in1_off+u*d1+c);
                                        Precision mixed=Precision(0);
                                        for(int vbegin=0;vbegin<mul2;
                                            vbegin+=stage_v) {
                                            const int width=Kokkos::min(
                                                stage_v,mul2-vbegin);
                                            for(int idx=tr;
                                                idx<tile_m*width;
                                                idx+=team.team_size()) {
                                                const int uu2=idx/width,
                                                    vv=idx%width;
                                                const int u2=u0+uu2;
                                                s(uu2*stride2+vv)=
                                                    (u2<mul1)
                                                    ?weight_view(w_off
                                                        +u2*mul2
                                                        +vbegin+vv)
                                                    :Precision(0);
                                            }
                                            team.team_barrier();
                                            if(active) {
                                                for(int v=0;v<width;++v) {
                                                    Precision pair=
                                                        Precision(0);
                                                    for(int c=0;c<d1;++c)
                                                        pair+=x1v[c]
                                                            *x2(node,
                                                                in2_off
                                                                    +(vbegin+v)
                                                                        *d1+c);
                                                    mixed+=pair
                                                        *inv_sqrt_d1
                                                        *s(uu*stride2+v);
                                                }
                                            }
                                            team.team_barrier();
                                        }
                                        if(active)
                                            output(node,out_off+u)
                                                +=pw*mixed;
                                    });
                            } else {
                            Kokkos::parallel_for("polar le uvu mode0",
                                Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                                    {0,0},{nodes,mul1}),
                                KOKKOS_LAMBDA(int node,int u) {
                                    Precision mixed=Precision(0);
                                    for(int v=0;v<mul2;++v) {
                                        Precision pair=Precision(0);
                                        for(int c=0;c<d1;++c)
                                            pair+=x1(node,in1_off+u*d1+c)
                                                *x2(node,in2_off+v*d1+c);
                                        mixed+=pair*inv_sqrt_d1
                                            *weight_view(w_off+u*mul2+v);
                                    }
                                    output(node,out_off+u)+=pw*mixed;
                                });
                            }
                        } else {
                            Kokkos::parallel_for("polar le uvu mode1",
                                Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                                    {0,0},{nodes,mul1}),
                                KOKKOS_LAMBDA(int node,int u) {
                                    Precision mixed=Precision(0);
                                    for(int v=0;v<mul2;++v)
                                        mixed+=x2(node,in2_off+v)
                                            *weight_view(w_off+u*mul2+v);
                                    const Precision contribution=
                                        pw*mixed*inv_sqrt_d1;
                                    for(int c=0;c<d1;++c)
                                        output(node,out_off+u*d1+c)+=
                                            contribution*x1(node,
                                                in1_off+u*d1+c);
                                });
                        }
                    }
                    Kokkos::parallel_for("polar le uvu mask",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                            {nodes,static_cast<int>(output.extent(1))}),
                        KOKKOS_LAMBDA(int node,int column) {
                            output(node,column)*=mask_view(column);
                        });
                };
                const auto le_to_host_vector=[&](const auto& view) {
                    const auto mirror=Kokkos::create_mirror_view_and_copy(
                        Kokkos::HostSpace(),view);
                    std::vector<typename decltype(mirror)::value_type>
                        values(mirror.size());
                    for(std::size_t i2=0;i2<values.size();++i2)
                        values[i2]=mirror.data()[i2];
                    return values;
                };
                const std::vector<int> le_q_meta=
                    le_to_host_vector(polar_le_q_meta);
                const std::vector<Precision> le_q_path_weight=
                    le_to_host_vector(polar_le_q_path_weight);
                const std::vector<int> le_v_meta=
                    le_to_host_vector(polar_le_v_meta);
                const std::vector<Precision> le_v_path_weight=
                    le_to_host_vector(polar_le_v_path_weight);
                auto node_feats=polar_node_feats_e3nn;
                launch_le_uvu(num_nodes,le_q_meta,le_q_path_weight,
                    polar_le_q_weight,polar_le_q_mask,node_feats,
                    polar_le_q_up,polar_le_dot_q);
                launch_le_uvu(num_nodes,le_v_meta,le_v_path_weight,
                    polar_le_v_weight,polar_le_v_mask,node_feats,
                    polar_le_v_up,polar_le_dot_v);
            }
            {
                auto invar=polar_le_invar,dot_q=polar_le_dot_q,dot_v=polar_le_dot_v;
                Kokkos::parallel_for("polar le invar concat",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                        {num_nodes,le_dot_width}),
                    KOKKOS_LAMBDA(int node,int column) {
                        invar(node,column)=dot_q(node,column);
                        invar(node,le_dot_width+column)=dot_v(node,column);
                    });
            }
            polar_le_mlp.evaluate(polar_le_invar,polar_le_mlp_out);
            {
                auto atom=polar_le_atom;
                auto mlp_out=polar_le_mlp_out;
                Kokkos::parallel_for("polar le atom readout",num_nodes,
                    KOKKOS_LAMBDA(int node) {
                        atom(node)=mlp_out(node,0);
                    });
            }
            // 总量：每图 scatter_sum（单图=整盒；无原子数缩放，mace L879-881）。
            const auto le_atom_host=Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(),polar_le_atom);
            Precision le_sum=Precision(0);
            for(int node=0;node<owned_nodes;++node)
                le_sum+=le_atom_host(node);
            {
                double global_le=static_cast<double>(le_sum);
                polar_global_sum_host(&global_le,1);
                le_sum=Precision(global_le);
            }
            polar_le_total=le_sum;
        }

        if(polar_enabled&&polar_le_enabled&&polar_le_add) {
            // 模块⑦ 能量装配（mace extensions.py L866-883）:
            //   total = e0 + inter_e（已含 scale/shift） + le_total
            // le 是 scale/shift 之后的加项（不乘 energy_scale），与 ⑥ electro 并列。
            auto energies=node_energies;
            auto le=polar_le_atom;
            Kokkos::parallel_for("polar le energy add",num_nodes,
                KOKKOS_LAMBDA(int node) {
                    energies(node)+=le(node);
                });
        }

        if(polar_enabled) {
            // P3 模块⑥ 能量装配:参考装配序 = (e0+inter_e) → +le → +electro
            // (extensions.py L866 → L925-929);scale_shift 仅作用于 node_inter_es
            // (L706-708),electro **未缩放** ⇒ 必须位于 "scale nonlinear energies"
            // 行之后,否则会被覆写为 e0+scale·E+shift(语义错)。本模型
            // scale=1.0/shift=0.0(JSON 实值)数值无差,顺序语义必须正确。
            // 单图口径: 整盒总量加入 node 0(calculator.py L3860 汇总全部 node 能量)。
            auto coulomb_energy=polar_coulomb_energy;
            auto coulomb_energies=node_energies;
            if(polar_global_rank_==0)
                Kokkos::parallel_for("add polar coulomb energy",1,
                    KOKKOS_LAMBDA(int) {
                        coulomb_energies(0)+=coulomb_energy(0);
                    });
        }
    }
    symmetrix_polar_profile_mark("pipeline:end");
}

template<typename Precision>
void MaceNonlinearKokkosT<Precision>::parse_polar_extension_tables()
{
    // [v2 适配] ⑥-B(module6 §2) + ⑦-4.2a(module7 §4.2a) 合并迁入(原 ctor 解析段):
    // 数据源=成员 polar_json(ctor 已赋值); epd 按主 tpp ctor ③ 段同一 JSON 路径重建;
    // 两段整体缩进 -4(函数体作用域); `pb.` 已替换为 `polar_json.`。
    const auto& epd=polar_json.at("electric_potential_descriptor");
    // ---- P3 模块⑥: coulomb_energy(GTO 表与 descriptor 同源;SI 表独立) ----
    // 依据: GL energy.py L33-68(同一 density_max_l/smearing_width/kspace_cutoff
    // 构造 density_basis / self_interaction / realspace 三子块);
    // 实测(2026-10-15): JSON 中 coulomb_energy.density_basis 与
    // electric_potential_descriptor.density_basis 全子树逐值相等。
    {
        const auto& cb=polar_json.at("coulomb_energy");
        polar_include_self_interaction=
            cb.at("include_self_interaction").get<bool>();
        polar_include_pbc_corrections=
            cb.at("include_pbc_corrections").get<bool>();
        // 逐值一致性断言(nlohmann::json == 为递归数值比较);不等即抛错,
        // 防止未来 extractor/模型失配时静默复用错误表。
        if(cb.at("density_basis")!=epd.at("density_basis")
            ||cb.at("kspace_cutoff").get<double>()
                !=epd.at("kspace_cutoff").get<double>())
            throw std::invalid_argument(
                "polar_blocks: coulomb_energy density_basis/kspace_cutoff must "
                "match electric_potential_descriptor (same-source GTO tables; "
                "see extract_polar_blocks.py::_gto_energy).");
        // SI 表 = coulomb 自己的 [4] 表(实测 overlap=[5.4160867,0.4011916×3],
        // select=[0,1,2,3]),与 descriptor 的 [8] 表**不同源**,且 realspace
        // 分支自相互作用符号约定与 k 空间相反 → 必须单独解析,不得复用。
        polar_coulomb_si_overlap=toKokkosView("polar coulomb si overlap",
            tensor_values<Precision>(cb.at("self_interaction_terms")
                .at("overlap_constants")));
        polar_coulomb_si_select=toKokkosView("polar coulomb si select",
            tensor_values<int>(cb.at("self_interaction_terms")
                .at("select_indices")));
        // [v3 适配·D-10] ⑥ 已落码(v2 起前向能量已接入 node_energies)按提案 §0-3/§10-2
        // "⑥ 落码时置位" 启用 coulomb 反向种子(hpp 默认 false 保留)。
        polar_coulomb_enabled=true;
    }

    // ---- P4 RBE: 随机批 Ewald 采样配置(可选; 依据 docs/design/RBE_P4_DESIGN.md) ----
    // JSON "rbe" 块: {enabled,p,sigma_q,seed};ENV 覆盖(MATOS_RBE_P/SEED/SIGMA)为
    // 门/调试测试钩子(生产以 JSON 为准,见 PLAN 依据账本)。缺省(无块且无 ENV)关闭
    // ⇒ 确定性路径逐位不变。
    if(polar_json.contains("rbe")
        ||std::getenv("MATOS_RBE_P")!=nullptr) {
        if(polar_json.contains("rbe")) {
            const auto& rbe=polar_json.at("rbe");
            polar_rbe_enabled=rbe.value("enabled",true);
            polar_rbe_p=rbe.value("p",200);
            polar_rbe_sigma=rbe.value("sigma_q",1.5);
            polar_rbe_seed=rbe.value("seed",20261016ULL);
        } else {
            polar_rbe_enabled=true;
        }
        if(const char* env=std::getenv("MATOS_RBE_P"))
            polar_rbe_p=std::atoi(env);
        if(const char* env=std::getenv("MATOS_RBE_SEED"))
            polar_rbe_seed=std::strtoull(env,nullptr,10);
        if(const char* env=std::getenv("MATOS_RBE_SIGMA"))
            polar_rbe_sigma=std::atof(env);
        if(polar_rbe_p<8||polar_rbe_p>65536)
            throw std::invalid_argument(
                "polar_blocks: rbe p must be in [8,65536].");
        if(polar_rbe_sigma<=0.0)
            throw std::invalid_argument(
                "polar_blocks: rbe sigma_q must be positive.");
    }

    // ---- P3 模块⑦: local_electron_energy（OneBodyMLPFieldReadout L652-752） ----
    // uvu 路径表列 = [in1_off,in2_off,out_off,w_off,mul1,mul2,mode,d1]
    // （与 PART1 parse_uvu_paths L51-87 同一语义；PART 为独立 TU，此处内联）。
    if(polar_json.contains("local_electron_energy")) {
        const auto& le=polar_json.at("local_electron_energy");
        polar_le_enabled=true;
        polar_le_add=polar_json.at("configs").value("add_local_electron_energy",false);
        polar_le_linear_up_q=E3LinearKokkosT<Precision>(le.at("linear_up_q"));
        polar_le_linear_up_v=E3LinearKokkosT<Precision>(le.at("linear_up_v"));
        const auto parse_le_uvu=[&](const nlohmann::json& data,
                Kokkos::View<int**,Kokkos::LayoutRight>& meta_view,
                Kokkos::View<Precision*>& paths_view,
                Kokkos::View<Precision*>& weight_view,
                Kokkos::View<Precision*>& mask_view,
                const char* meta_label,const char* paths_label,
                const char* weight_label,const char* mask_label) {
            const Irreps in1(data.at("irreps_in1").get<std::string>());
            const Irreps in2(data.at("irreps_in2").get<std::string>());
            const Irreps out(data.at("irreps_out").get<std::string>());
            const auto weights=tensor_values<Precision>(data.at("weight"));
            const auto mask=tensor_values<Precision>(data.at("output_mask"));
            std::vector<int> meta;
            std::vector<Precision> paths;
            int weight_offset=0;
            for(const auto& value:data.at("instructions")) {
                const auto& a=in1.blocks.at(value.at("i_in1").get<int>());
                const auto& b=in2.blocks.at(value.at("i_in2").get<int>());
                const auto& c=out.blocks.at(value.at("i_out").get<int>());
                const auto shape=value.at("path_shape")
                    .get<std::vector<int>>();
                const int d1=2*a.l+1;
                int mode=0;
                if(c.l!=0) {
                    mode=1;
                    if(b.l!=0||a.l!=c.l)
                        throw std::invalid_argument(
                            "polar le uvu: mode1 requires l2==0 and l1==l_out.");
                } else if(a.l!=b.l)
                    throw std::invalid_argument(
                        "polar le uvu: mode0 requires l1==l2.");
                meta.insert(meta.end(),{a.offset,b.offset,c.offset,
                    weight_offset,a.multiplicity,b.multiplicity,mode,d1});
                paths.push_back(value.at("path_weight").get<Precision>());
                weight_offset+=static_cast<int>(shape[0])
                    *static_cast<int>(shape[1]);
            }
            if(weight_offset!=static_cast<int>(weights.size()))
                throw std::invalid_argument(
                    "polar le uvu: weight size mismatch.");
            meta_view=toKokkosView(meta_label,meta,
                static_cast<int>(paths.size()),8);
            paths_view=toKokkosView(paths_label,paths);
            weight_view=toKokkosView(weight_label,weights);
            mask_view=toKokkosView(mask_label,mask);
        };
        parse_le_uvu(le.at("dot_products_q"),polar_le_q_meta,
            polar_le_q_path_weight,polar_le_q_weight,polar_le_q_mask,
            "polar le q meta","polar le q path weight",
            "polar le q weight","polar le q mask");
        parse_le_uvu(le.at("dot_products_v"),polar_le_v_meta,
            polar_le_v_path_weight,polar_le_v_weight,polar_le_v_mask,
            "polar le v meta","polar le v path weight",
            "polar le v weight","polar le v mask");
        // RadialMLP = [Linear,LayerNorm,SiLU]×3 + Linear（radial.py L361-384）。
        // 提取器当前以 "layernorm" 落盘；引擎 AffineMLP schema 为 "layer_norm"
        // （affine_mlp.cpp L57）→ 归一（两种拼写均接受；"layer_norm" 原样）。
        nlohmann::json mlp=le.at("mlp");
        for(auto& layer:mlp.at("layers"))
            if(layer.value("type",std::string())=="layernorm")
                layer["type"]="layer_norm";
        polar_le_mlp=AffineMLPKokkosT<Precision>(mlp);
        // 尺寸契约（装载即校验，失败信息见模块名）。
        if(polar_le_linear_up_q.input_dimension()!=8
            ||polar_le_linear_up_v.input_dimension()!=16)
            throw std::invalid_argument(
                "polar_blocks: local_electron_energy linear_up inputs must be 8/16.");
        if(polar_le_linear_up_q.output_dimension()!=2048
            ||polar_le_linear_up_v.output_dimension()!=2048)
            throw std::invalid_argument(
                "polar_blocks: local_electron_energy linear_up outputs must be 2048.");
        if(static_cast<int>(polar_le_q_mask.extent(0))!=512
            ||static_cast<int>(polar_le_v_mask.extent(0))!=512)
            throw std::invalid_argument(
                "polar_blocks: local_electron_energy dot_products outputs must be 512.");
        if(polar_le_mlp.input_size()!=1024||polar_le_mlp.output_size()!=1)
            throw std::invalid_argument(
                "polar_blocks: local_electron_energy mlp must map 1024 -> 1.");
    } else polar_le_enabled=false;
}

// [v3 适配] 本块按适配规则 1 整体迁入 polar TU(mace_polar_impl.tpp; 显式实例化之前)。
// 相对 module8_patch_final.md §4.2 原文的偏离(全表见 docs/merge_adapt_8.md §2):
//   D-04 d_in 步链: polar_scf_step_densities_post → 既有 polar_scf_step_densities(校后口径同值);
//   D-05 ⑥ 种子密度源: polar_spin_charge_density → polar_scf_final_density(与 ⑥ 前向同源);
//   D-06 lambda 形参 Kokkos::View<...**> 补 LayoutRight(bench/polar_lint.py R3);
//   D-08 fukui_reverse 三处系数按引擎前向闭式伴随修正(FD 证据见脚本 [8] 自检)。
// =====================================================================================
// P3 模块⑧: 极性链全链反向(力)。
// 链序: 种子(⑥/⑦) → SCF 逐步(replay→Fukui 校正逆→更新块逆→描述符逆) → 初始化 Fukui 逆
//       → lr_src 逆 / mixer 逆 → ⑧-A 注入 layer_adjoint(init=false) → polar_node_forces。
// 数学逐项对照: docs/POLAR_REVERSE_DESIGN.md §2; 前向锚: mace extensions.py L700-935,
// graph_longrange features.py L28-135 / energy.py L17-30; 引擎前向: 本文件 P3 模块①②③④⑤ 段。
// =====================================================================================
template<typename Precision>
void MaceNonlinearKokkosT<Precision>::compute_polar_reverse_forces(
    const int num_nodes)
{
    if(!polar_enabled) return;
    const int owned_nodes=polar_owned_node_count(num_nodes);
    constexpr double two_pi=6.283185307179586476925286766559;
    constexpr double two_pi_cubed=two_pi*two_pi*two_pi;
    constexpr double polar_field_constant=1.0/(5.526349406e-3);
    const int layers=static_cast<int>(interactions.size());
    const int num_k=static_cast<int>(polar_k_vectors.extent(0));
    const int feature_width=polar_feature_width;
    if(static_cast<int>(polar_cosines.extent(0))!=num_k
        ||static_cast<int>(polar_cosines.extent(1))!=num_nodes
        ||static_cast<int>(polar_k_vectors.extent(0))!=num_k||num_k==0)
        throw std::runtime_error(
            "polar reverse: geometry cache missing or stale "
            "(run the polar forward first).");

        symmetrix_polar_profile_mark("rev0:clear");
// ---------- 0) 累加器清零 ----------
    ensure_view(polar_node_forces,polar_node_forces_storage,
        3*static_cast<std::size_t>(num_nodes));
    ordered_kokkos_deep_copy(polar_node_forces,0.0);
    ensure_view(polar_cell_adjoint,polar_cell_adjoint_storage,9);
    ordered_kokkos_deep_copy(polar_cell_adjoint,Precision(0));
    ensure_view(polar_rev_density_adjoint,polar_rev_density_adjoint_storage,
        num_nodes,8);
    ensure_view(polar_rev_position_adjoint,polar_rev_position_adjoint_storage,
        num_nodes,3);
    ordered_kokkos_deep_copy(polar_rev_density_adjoint,Precision(0));
    ordered_kokkos_deep_copy(polar_rev_position_adjoint,Precision(0));

        symmetrix_polar_profile_mark("rev1:lambda-setup");
// ---------- 1) 公共小工具(纯 λ,无副作用;先定义后使用) ----------
    // (2a) ⑧-A: e3nn 基伴 → 内部基伴,散射累加(单射,atomic 自卫;见 §1.2-4)。
    const auto basis_adjoint_add=[&](Kokkos::View<const Precision**,Kokkos::LayoutRight> e3nn_adj,
            Kokkos::View<Precision**,Kokkos::LayoutRight> internal_adj) {
        if(static_cast<int>(e3nn_adj.extent(1))!=feature_width
            ||static_cast<int>(internal_adj.extent(1))!=feature_width)
            throw std::runtime_error(
                "polar reverse: node-feature adjoint width mismatch.");
        auto source=polar_basis_source;
        auto sign=polar_basis_sign;
        Kokkos::parallel_for("polar basis adjoint scatter",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                {num_nodes,feature_width}),
            KOKKOS_LAMBDA(int node,int column) {
                // e3nn 列 c = sign_c·内部列 source(c); c↦source(c) 为分块双射
                // ⇒ 逐 (node,j) 唯一写者;atomic 仅为鲁棒。
                Kokkos::atomic_add(&internal_adj(node,source(column)),
                    sign(column)*e3nn_adj(node,column));
            });
    };
    // (2b) 逐通道 Fukui 校正反向(初始与逐步同构; §1.2-5)。
    //      scalars_host=[S_rawα,S_rawβ,predα,predβ,Fα,Fβ];sources(N,2) 从 source_offset 列起;
    //      s_bar(N,2) 输出;dbar 就地 += pred̄(l=0 列)。
    //      transform 槽位=[1,pred̄α,1,pred̄β,term2α,term2β,Fα,Fβ](索引 0..7;
    //      F=factor_σ=(T−pred)/S_used; [v3 修正·D-08] 见核内注释)。
    const auto fukui_reverse=[&](Kokkos::View<Precision**,Kokkos::LayoutRight> dbar,
            Kokkos::View<const Precision**,Kokkos::LayoutRight> sources,const int source_offset,
            const std::array<Precision,6>& scalars_host,
            Kokkos::View<Precision**,Kokkos::LayoutRight> s_bar) {
        const auto dbar_host=Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),dbar);
        const auto sources_host=Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),sources);
        ensure_view(polar_rev_transform,polar_rev_transform_storage,8);
        auto transform=Kokkos::create_mirror_view(
            Kokkos::HostSpace(),polar_rev_transform);
        double fbar_values[2]={0.0,0.0};
        for(int channel=0;channel<2;++channel) {
            const Precision s_raw=scalars_host[channel];
            const Precision s_used=(s_raw==Precision(0))?Precision(1):s_raw;
            double fbar=0.0;                                  // F̄ = Σ_n s̃·d̄(l=0)
            for(int node=0;node<owned_nodes;++node)
                fbar+=static_cast<double>(dbar_host(node,4*channel))
                    *(static_cast<double>(sources_host(node,source_offset+channel))
                        /static_cast<double>(s_used));
            fbar_values[channel]=fbar;
        }
        polar_global_sum_host(fbar_values,2);
        for(int channel=0;channel<2;++channel) {
            const Precision s_raw=scalars_host[channel];
            const bool clamped=(s_raw==Precision(0));
            const Precision f=scalars_host[4+channel];
            const double fbar=fbar_values[channel];
            // [v3 修正·D-08] 引擎/mace 前向: d_out(n)=d_pre(n)+s_n·f, f=(T−pred)/S_used
            // (单一 1/S; mace extensions.py L742-766 与 ⑤ 前向核)。闭式伴随:
            //   pred̄=−F̄;  s̄=f·d̄ −[未钳制] f·F̄;  F̄=Σ_n (s_n/S_used)·d̄(n,l=0)。
            // 提案按设计 §2 ⑧-2 的 s̃·F(F=(T−pred)/S) 参数化推出, 对引擎前向多含 1/S,
            // 第二项系数应为 −f·F̄(非 −2·f·F̄/S_raw); FD 证据: merge_adapt_8.md §2-D08。
            transform(2*channel)=Precision(1);
            transform(2*channel+1)=Precision(-fbar);
            transform(4+channel)=clamped?Precision(0)
                :Precision(-fbar*static_cast<double>(f));
            transform(6+channel)=f;
        }
        Kokkos::deep_copy(polar_rev_transform,transform);
        auto dbar_l=dbar;auto s_bar_l=s_bar;auto tr=polar_rev_transform;
        Kokkos::parallel_for("polar reverse fukui correction",num_nodes,
            KOKKOS_LAMBDA(int node) {
                for(int channel=0;channel<2;++channel) {
                    const int column=4*channel;
                    s_bar_l(node,channel)=tr(6+channel)*tr(2*channel)
                        *dbar_l(node,column)+tr(4+channel);
                    dbar_l(node,column)+=tr(2*channel+1);
                }
            });
    };
    // (2c) uvu 反向(mode0/mode1; 语义=⑤前向 launch_uvu 的逐路径伴随,设计 §2 ⑧-4b)。
    const auto uvu_reverse=[&](const std::vector<int>& meta,
            const std::vector<Precision>& path_weight,
            Kokkos::View<Precision*> weight_view,Kokkos::View<Precision*> mask_view,
            Kokkos::View<const Precision**,Kokkos::LayoutRight> x1,Kokkos::View<const Precision**,Kokkos::LayoutRight> x2,
            Kokkos::View<const Precision**,Kokkos::LayoutRight> out_adj,
            Kokkos::View<Precision**,Kokkos::LayoutRight> x1_adj,Kokkos::View<Precision**,Kokkos::LayoutRight> x2_adj) {
        const int paths=static_cast<int>(meta.size())/8;
        for(int p=0;p<paths;++p) {
            const int in1_off=meta[p*8+0],in2_off=meta[p*8+1],
                out_off=meta[p*8+2],w_off=meta[p*8+3],
                mul1=meta[p*8+4],mul2=meta[p*8+5],mode=meta[p*8+6],
                d1=meta[p*8+7];
            const Precision pw=path_weight[p];
            const Precision inv_sqrt_d1=Precision(1)
                /Kokkos::sqrt(Precision(d1));
            // mixed_bar(node,u) is loop-invariant in the x2 (node,v)
            // reduction.  Cache the exact scalar from the x1 pass and reuse
            // it in x2: removes two loads + a multiply from the inner u loop
            // without changing any accumulation order (bitwise identical).
            ensure_view(polar_rev_uvu_mixed_bar,
                polar_rev_uvu_mixed_bar_storage,num_nodes,mul1);
            auto mixed_bar_view=polar_rev_uvu_mixed_bar;
            if(mode==0) {
                if(d1<=8) {
                    // Shared-memory weight tile version of the mode-0
                    // adjoints.  Round-3 adds a deterministic split of the
                    // reduction dimension (MATOS_UVU_REV_SWEEP, default 4):
                    // each team owns one v- (x1) or u- (x2) chunk, writes a
                    // per-chunk partial, and a second kernel folds the
                    // partials in ascending chunk order.
                    const char* sweep_env=std::getenv(
                        "MATOS_UVU_REV_SWEEP");
                    int sweep=4;
                    if(sweep_env!=nullptr) sweep=std::atoi(sweep_env);
                    if(sweep!=2&&sweep!=4&&sweep!=8) sweep=4;
                    int tile_n=32, tile_m=32;
                    uvu_tile_shape(tile_n,tile_m);
                    if(sweep<=1) {
                        // Single-pass branch stages the whole weight seam in
                        // shared memory; keep that tile inside a 32 KB budget.
                        while(tile_m>1&&static_cast<std::size_t>(tile_m)
                                *(mul2+1)*sizeof(Precision)>32768)
                            tile_m/=2;
                    }
                    const int n_tiles=(num_nodes+tile_n-1)/tile_n;
                    const int m1_tiles=(mul1+tile_m-1)/tile_m;
                    const int m2_tiles=(mul2+tile_m-1)/tile_m;
                    using scratch_view=Kokkos::View<Precision*,
                        Kokkos::DefaultExecutionSpace::scratch_memory_space,
                        Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
                    if(sweep>1) {
                        const int chunks=sweep;
                        const int x1_chunk=(mul2+chunks-1)/chunks;
                        const int x2_chunk=(mul1+chunks-1)/chunks;
                        const int x1_stride=x1_chunk+1;
                        const int x2_stride=x2_chunk+1;
                        const std::size_t x1_smem=
                            scratch_view::shmem_size(tile_m*x1_stride);
                        const std::size_t x2_smem=
                            scratch_view::shmem_size(tile_m*x2_stride);
                        ensure_view(polar_rev_uvu_partial_x1,
                            polar_rev_uvu_partial_x1_storage,chunks,
                            num_nodes,mul1*d1);
                        ensure_view(polar_rev_uvu_partial_x2,
                            polar_rev_uvu_partial_x2_storage,chunks,
                            num_nodes,mul2*d1);
                        auto partial_x1=polar_rev_uvu_partial_x1;
                        auto partial_x2=polar_rev_uvu_partial_x2;
                        Kokkos::parallel_for("polar reverse uvu mode0 x1",
                            Kokkos::TeamPolicy<>(
                                n_tiles*m1_tiles*chunks,tile_n*tile_m)
                                .set_scratch_size(0,
                                    Kokkos::PerTeam(x1_smem)),
                            KOKKOS_LAMBDA(
                                const Kokkos::TeamPolicy<>::member_type&
                                    team) {
                                const int tile=team.league_rank();
                                const int chunk_index=tile%chunks;
                                const int block=tile/chunks;
                                const int nt=block/m1_tiles,
                                    ut=block%m1_tiles;
                                const int n0=nt*tile_n,u0=ut*tile_m;
                                const int v_begin=chunk_index*x1_chunk;
                                const int v_end=Kokkos::min(
                                    mul2,v_begin+x1_chunk);
                                const int chunk_width=v_end-v_begin;
                                scratch_view s(team.team_scratch(0),
                                    tile_m*x1_stride);
                                for(int idx=team.team_rank();
                                    idx<tile_m*chunk_width;
                                    idx+=team.team_size()) {
                                    const int uu=idx/chunk_width,
                                        vv=idx%chunk_width;
                                    const int u=u0+uu;
                                    s(uu*x1_stride+vv)=(u<mul1)
                                        ?weight_view(w_off+u*mul2
                                            +v_begin+vv)
                                        :Precision(0);
                                }
                                team.team_barrier();
                                const int tr=team.team_rank();
                                const int node=n0+tr/tile_m,
                                    uu=tr%tile_m;
                                if(node<num_nodes&&u0+uu<mul1) {
                                    const int u=u0+uu;
                                    const Precision mixed_bar=pw
                                        *out_adj(node,out_off+u)
                                        *mask_view(out_off+u);
                                    if(chunk_index==0)
                                        mixed_bar_view(node,u)=mixed_bar;
                                    Precision acc[8];
                                    for(int c=0;c<d1;++c)
                                        acc[c]=Precision(0);
                                    for(int v=v_begin;v<v_end;++v) {
                                        const Precision pair_bar=mixed_bar
                                            *inv_sqrt_d1
                                            *s(uu*x1_stride+v-v_begin);
                                        for(int c=0;c<d1;++c)
                                            acc[c]+=pair_bar
                                                *x2(node,in2_off+v*d1+c);
                                    }
                                    for(int c=0;c<d1;++c)
                                        partial_x1(chunk_index,node,
                                            u*d1+c)=acc[c];
                                }
                            });
                        Kokkos::parallel_for(
                            "polar reverse uvu mode0 x1 reduce",
                            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                                {0,0},{num_nodes,mul1*d1}),
                            KOKKOS_LAMBDA(int node,int column) {
                                Precision value=Precision(0);
                                for(int chunk_index=0;
                                    chunk_index<chunks;++chunk_index)
                                    value+=partial_x1(
                                        chunk_index,node,column);
                                x1_adj(node,in1_off+column)+=value;
                            });
                        const char* mb_env=std::getenv(
                            "MATOS_UVU_X2_MB_SMEM");
                        const bool mb_smem=!(mb_env!=nullptr
                            &&std::atoi(mb_env)==0);
                        if(mb_smem) {
                            // Stage a (64 nodes x 32 u) mixed_bar tile and a
                            // (4 v x 32 u) weight tile in shared memory.
                            // Four v threads share one node's mixed_bar row,
                            // so this cuts those global loads 4x and keeps
                            // the ascending-u accumulation order.
                            constexpr int x2_sub=32;
                            const int sub_smem_elems=
                                (tile_m+tile_n)*x2_sub;
                            const std::size_t sub_smem=
                                scratch_view::shmem_size(sub_smem_elems);
                            Kokkos::parallel_for(
                                "polar reverse uvu mode0 x2",
                                Kokkos::TeamPolicy<>(
                                    n_tiles*m2_tiles*chunks,
                                    tile_n*tile_m)
                                    .set_scratch_size(0,
                                        Kokkos::PerTeam(sub_smem)),
                                KOKKOS_LAMBDA(
                                    const Kokkos::TeamPolicy<>::
                                        member_type& team) {
                                    const int tile=team.league_rank();
                                    const int chunk_index=tile%chunks;
                                    const int block=tile/chunks;
                                    const int nt=block/m2_tiles,
                                        vt=block%m2_tiles;
                                    const int n0=nt*tile_n,
                                        v0=vt*tile_m;
                                    const int u_begin=
                                        chunk_index*x2_chunk;
                                    const int u_end=Kokkos::min(
                                        mul1,u_begin+x2_chunk);
                                    scratch_view scratch(
                                        team.team_scratch(0),
                                        sub_smem_elems);
                                    scratch_view sw(
                                        scratch.data(),
                                        tile_m*x2_sub);
                                    scratch_view smb(
                                        scratch.data()
                                            +tile_m*x2_sub,
                                        tile_n*x2_sub);
                                    const int tr=team.team_rank();
                                    const int node=n0+tr/tile_m,
                                        nn=tr/tile_m,vv=tr%tile_m;
                                    Precision acc[8];
                                    for(int c=0;c<d1;++c)
                                        acc[c]=Precision(0);
                                    for(int u0=u_begin;u0<u_end;
                                        u0+=x2_sub) {
                                        const int uw=Kokkos::min(
                                            x2_sub,u_end-u0);
                                        for(int idx=tr;
                                            idx<tile_m*uw;
                                            idx+=team.team_size()) {
                                            const int vvi=idx/uw,
                                                uu=idx%uw;
                                            const int v=v0+vvi;
                                            sw(vvi*uw+uu)=(v<mul2)
                                                ?weight_view(w_off
                                                    +(u0+uu)*mul2+v)
                                                :Precision(0);
                                        }
                                        for(int idx=tr;
                                            idx<tile_n*uw;
                                            idx+=team.team_size()) {
                                            const int ni=idx/uw,
                                                uu=idx%uw;
                                            smb(ni*uw+uu)=
                                                (n0+ni<num_nodes)
                                                ?mixed_bar_view(
                                                    n0+ni,u0+uu)
                                                :Precision(0);
                                        }
                                        team.team_barrier();
                                        if(node<num_nodes
                                            &&v0+vv<mul2) {
                                            for(int uu=0;uu<uw;++uu) {
                                                const Precision pair_bar=
                                                    smb(nn*uw+uu)
                                                    *inv_sqrt_d1
                                                    *sw(vv*uw+uu);
                                                for(int c=0;c<d1;++c)
                                                    acc[c]+=pair_bar
                                                        *x1(node,in1_off
                                                            +(u0+uu)*d1
                                                            +c);
                                            }
                                        }
                                        team.team_barrier();
                                    }
                                    if(node<num_nodes&&v0+vv<mul2) {
                                        const int v=v0+vv;
                                        for(int c=0;c<d1;++c)
                                            partial_x2(chunk_index,
                                                node,v*d1+c)=acc[c];
                                    }
                                });
                        } else {
                        Kokkos::parallel_for("polar reverse uvu mode0 x2",
                            Kokkos::TeamPolicy<>(
                                n_tiles*m2_tiles*chunks,tile_n*tile_m)
                                .set_scratch_size(0,
                                    Kokkos::PerTeam(x2_smem)),
                            KOKKOS_LAMBDA(
                                const Kokkos::TeamPolicy<>::member_type&
                                    team) {
                                const int tile=team.league_rank();
                                const int chunk_index=tile%chunks;
                                const int block=tile/chunks;
                                const int nt=block/m2_tiles,
                                    vt=block%m2_tiles;
                                const int n0=nt*tile_n,v0=vt*tile_m;
                                const int u_begin=chunk_index*x2_chunk;
                                const int u_end=Kokkos::min(
                                    mul1,u_begin+x2_chunk);
                                const int chunk_width=u_end-u_begin;
                                scratch_view s(team.team_scratch(0),
                                    tile_m*x2_stride);
                                for(int idx=team.team_rank();
                                    idx<tile_m*chunk_width;
                                    idx+=team.team_size()) {
                                    const int vv=idx/chunk_width,
                                        uu=idx%chunk_width;
                                    const int v=v0+vv;
                                    s(vv*x2_stride+uu)=(v<mul2)
                                        ?weight_view(w_off
                                            +(u_begin+uu)*mul2+v)
                                        :Precision(0);
                                }
                                team.team_barrier();
                                const int tr=team.team_rank();
                                const int node=n0+tr/tile_m,
                                    vv=tr%tile_m;
                                if(node<num_nodes&&v0+vv<mul2) {
                                    const int v=v0+vv;
                                    Precision acc[8];
                                    for(int c=0;c<d1;++c)
                                        acc[c]=Precision(0);
                                    for(int u=u_begin;u<u_end;++u) {
                                        const Precision pair_bar=
                                            mixed_bar_view(node,u)
                                            *inv_sqrt_d1
                                            *s(vv*x2_stride+u-u_begin);
                                        for(int c=0;c<d1;++c)
                                            acc[c]+=pair_bar
                                                *x1(node,in1_off
                                                    +u*d1+c);
                                    }
                                    for(int c=0;c<d1;++c)
                                        partial_x2(chunk_index,node,
                                            v*d1+c)=acc[c];
                                }
                            });
                        }
                        Kokkos::parallel_for(
                            "polar reverse uvu mode0 x2 reduce",
                            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                                {0,0},{num_nodes,mul2*d1}),
                            KOKKOS_LAMBDA(int node,int column) {
                                Precision value=Precision(0);
                                for(int chunk_index=0;
                                    chunk_index<chunks;++chunk_index)
                                    value+=partial_x2(
                                        chunk_index,node,column);
                                x2_adj(node,in2_off+column)+=value;
                            });
                    } else {
                    const int stride2=mul2+1,stride1=mul1+1;
                    const std::size_t x1_smem=scratch_view::shmem_size(
                        tile_m*stride2);
                    const std::size_t x2_smem=scratch_view::shmem_size(
                        tile_m*stride1);
                    Kokkos::parallel_for("polar reverse uvu mode0 x1",
                        Kokkos::TeamPolicy<>(n_tiles*m1_tiles,tile_n*tile_m)
                            .set_scratch_size(0,Kokkos::PerTeam(x1_smem)),
                        KOKKOS_LAMBDA(
                            const Kokkos::TeamPolicy<>::member_type& team) {
                            const int tile=team.league_rank();
                            const int nt=tile/m1_tiles,ut=tile%m1_tiles;
                            const int n0=nt*tile_n,u0=ut*tile_m;
                            scratch_view s(team.team_scratch(0),
                                tile_m*stride2);
                            for(int idx=team.team_rank();idx<tile_m*mul2;
                                idx+=team.team_size()) {
                                const int uu=idx/mul2,vv=idx%mul2;
                                const int u=u0+uu;
                                s(uu*stride2+vv)=(u<mul1)
                                    ?weight_view(w_off+u*mul2+vv)
                                    :Precision(0);
                            }
                            team.team_barrier();
                            const int tr=team.team_rank();
                            const int node=n0+tr/tile_m,uu=tr%tile_m;
                            if(node<num_nodes&&u0+uu<mul1) {
                                const int u=u0+uu;
                                const Precision mixed_bar=pw
                                    *out_adj(node,out_off+u)
                                    *mask_view(out_off+u);
                                mixed_bar_view(node,u)=mixed_bar;
                                Precision acc[8];
                                for(int c=0;c<d1;++c) acc[c]=Precision(0);
                                for(int v=0;v<mul2;++v) {
                                    const Precision pair_bar=mixed_bar
                                        *inv_sqrt_d1*s(uu*stride2+v);
                                    for(int c=0;c<d1;++c)
                                        acc[c]+=pair_bar
                                            *x2(node,in2_off+v*d1+c);
                                }
                                for(int c=0;c<d1;++c)
                                    x1_adj(node,in1_off+u*d1+c)+=acc[c];
                            }
                        });
                    Kokkos::parallel_for("polar reverse uvu mode0 x2",
                        Kokkos::TeamPolicy<>(n_tiles*m2_tiles,tile_n*tile_m)
                            .set_scratch_size(0,Kokkos::PerTeam(x2_smem)),
                        KOKKOS_LAMBDA(
                            const Kokkos::TeamPolicy<>::member_type& team) {
                            const int tile=team.league_rank();
                            const int nt=tile/m2_tiles,vt=tile%m2_tiles;
                            const int n0=nt*tile_n,v0=vt*tile_m;
                            scratch_view s(team.team_scratch(0),
                                tile_m*stride1);
                            for(int idx=team.team_rank();idx<tile_m*mul1;
                                idx+=team.team_size()) {
                                const int vv=idx/mul1,u=idx%mul1;
                                s(vv*stride1+u)=(v0+vv<mul2)
                                    ?weight_view(w_off+u*mul2+v0+vv)
                                    :Precision(0);
                            }
                            team.team_barrier();
                            const int tr=team.team_rank();
                            const int node=n0+tr/tile_m,vv=tr%tile_m;
                            if(node<num_nodes&&v0+vv<mul2) {
                                const int v=v0+vv;
                                Precision acc[8];
                                for(int c=0;c<d1;++c) acc[c]=Precision(0);
                                for(int u=0;u<mul1;++u) {
                                    const Precision pair_bar=mixed_bar_view(
                                        node,u)*inv_sqrt_d1*s(vv*stride1+u);
                                    for(int c=0;c<d1;++c)
                                        acc[c]+=pair_bar
                                            *x1(node,in1_off+u*d1+c);
                                }
                                for(int c=0;c<d1;++c)
                                    x2_adj(node,in2_off+v*d1+c)+=acc[c];
                            }
                        });
                    }
                } else {
                    Kokkos::parallel_for("polar reverse uvu mode0 x1",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                            {0,0},{num_nodes,mul1}),
                        KOKKOS_LAMBDA(int node,int u) {
                            const Precision mixed_bar=pw*out_adj(node,out_off+u)
                                *mask_view(out_off+u);
                            mixed_bar_view(node,u)=mixed_bar;
                            for(int c=0;c<d1;++c) {
                                Precision acc=Precision(0);
                                for(int v=0;v<mul2;++v) {
                                    const Precision pair_bar=mixed_bar
                                        *inv_sqrt_d1
                                        *weight_view(w_off+u*mul2+v);
                                    acc+=pair_bar*x2(node,in2_off+v*d1+c);
                                }
                                x1_adj(node,in1_off+u*d1+c)+=acc;
                            }
                        });
                    Kokkos::parallel_for("polar reverse uvu mode0 x2",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                            {0,0},{num_nodes,mul2}),
                        KOKKOS_LAMBDA(int node,int v) {
                            for(int c=0;c<d1;++c) {
                                Precision acc=Precision(0);
                                for(int u=0;u<mul1;++u) {
                                    const Precision mixed_bar=
                                        mixed_bar_view(node,u);
                                    const Precision pair_bar=mixed_bar
                                        *inv_sqrt_d1
                                        *weight_view(w_off+u*mul2+v);
                                    acc+=pair_bar*x1(node,in1_off+u*d1+c);
                                }
                                x2_adj(node,in2_off+v*d1+c)+=acc;
                            }
                        });
                }
            } else {
                Kokkos::parallel_for("polar reverse uvu mode1 x1",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                        {0,0},{num_nodes,mul1}),
                    KOKKOS_LAMBDA(int node,int u) {
                        Precision mixed=Precision(0);
                        for(int v=0;v<mul2;++v)
                            mixed+=x2(node,in2_off+v)
                                *weight_view(w_off+u*mul2+v);
                        const Precision contribution=pw*mixed*inv_sqrt_d1;
                        Precision mixed_bar=Precision(0);
                        for(int c=0;c<d1;++c) {
                            const int i1=in1_off+u*d1+c;
                            const Precision o=out_adj(node,out_off+u*d1+c)
                                *mask_view(out_off+u*d1+c);
                            x1_adj(node,i1)+=contribution*o;
                            mixed_bar+=x1(node,i1)*o;
                        }
                        mixed_bar_view(node,u)=mixed_bar;
                    });
                Kokkos::parallel_for("polar reverse uvu mode1 x2",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                        {0,0},{num_nodes,mul2}),
                    KOKKOS_LAMBDA(int node,int v) {
                        Precision acc=Precision(0);
                        for(int u=0;u<mul1;++u)
                            acc+=pw*inv_sqrt_d1*mixed_bar_view(node,u)
                                *weight_view(w_off+u*mul2+v);
                        x2_adj(node,in2_off+v)+=acc;
                    });
            }
        }
    };
    // (2d) 模块⑤内联 Gate 的反向(scalars=cst_s·silu;gates=cst_g·sigmoid⊗gated)。
    const auto readout_gate_reverse=[&](Kokkos::View<const Precision**,Kokkos::LayoutRight> src,
            Kokkos::View<const Precision**,Kokkos::LayoutRight> dst_adj,
            Kokkos::View<Precision**,Kokkos::LayoutRight> src_adj,
            const Precision scalar_cst,const Precision gate_cst,
            const int scalars,const int gates) {
        Kokkos::parallel_for("polar reverse readout gate",num_nodes,
            KOKKOS_LAMBDA(int node) {
                for(int i=0;i<scalars;++i) {
                    const Precision s=src(node,i);
                    const Precision sig=Precision(1)
                        /(Precision(1)+Kokkos::exp(-s));
                    src_adj(node,i)+=dst_adj(node,i)*scalar_cst
                        *(sig+s*sig*(Precision(1)-sig));
                }
                for(int i=0;i<gates;++i) {
                    const Precision gs=src(node,scalars+i);
                    const Precision sig=Precision(1)
                        /(Precision(1)+Kokkos::exp(-gs));
                    const Precision g=gate_cst*sig;
                    Precision gate_bar=Precision(0);
                    for(int c=0;c<3;++c) {
                        const int i3=scalars+3*i+c;
                        const int iv=scalars+gates+3*i+c;
                        src_adj(node,iv)+=g*dst_adj(node,i3);
                        gate_bar+=src(node,iv)*dst_adj(node,i3);
                    }
                    src_adj(node,scalars+i)+=gate_bar*gate_cst*sig
                        *(Precision(1)-sig);
                }
            });
    };
    // (2e) LayerNorm+SiLU 反向: dx_c = inv·γ_c·dz_c − inv/W·Σγdz
    //      − inv³·y_c/W·Σ(γdz·y);dz = ā·σ(z)(1+z(1−σ(z))),z=LN(h),a=silu(z)。
    const auto ln_silu_reverse=[&](Kokkos::View<const Precision**,Kokkos::LayoutRight> h,
            Kokkos::View<const Precision**,Kokkos::LayoutRight> a,
            Kokkos::View<const Precision**,Kokkos::LayoutRight> a_adj,
            Kokkos::View<Precision**,Kokkos::LayoutRight> h_adj,
            Kokkos::View<const Precision*> gamma,
            Kokkos::View<const Precision*> beta,
            const Precision eps) {
        const int width=static_cast<int>(h.extent(1));
        auto h_l=h;auto a_adj_l=a_adj;auto h_adj_l=h_adj;auto g=gamma;auto b=beta;
        const int ln_team=width<128?width:128;
        using scratch_view=Kokkos::View<Precision*,
            Kokkos::DefaultExecutionSpace::scratch_memory_space,
            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        const std::size_t smem=scratch_view::shmem_size(2*width);
        Kokkos::parallel_for("polar reverse layernorm silu",
            Kokkos::TeamPolicy<>(num_nodes,ln_team)
                .set_scratch_size(0,Kokkos::PerTeam(smem)),
            KOKKOS_LAMBDA(
                const Kokkos::TeamPolicy<>::member_type& team) {
                const int node=team.league_rank();
                Precision mean=Precision(0);
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team,width),
                    [&](const int c,Precision& sum) {
                        sum+=h_l(node,c);
                    },mean);
                mean/=Precision(width);
                Precision var=Precision(0);
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team,width),
                    [&](const int c,Precision& sum) {
                        const Precision delta=h_l(node,c)-mean;
                        sum+=delta*delta;
                    },var);
                var/=Precision(width);
                const Precision inv=Precision(1)/Kokkos::sqrt(var+eps);
                scratch_view scratch(team.team_scratch(0),2*width);
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team,width),
                    [&](const int c) {
                        const Precision y=h_l(node,c)-mean;
                        const Precision z=y*inv*g(c)+b(c);
                        const Precision sig=Precision(1)
                            /(Precision(1)+Kokkos::exp(-z));
                        const Precision dz=a_adj_l(node,c)*sig
                            *(Precision(1)+z*(Precision(1)-sig));
                        scratch(c)=y;
                        scratch(width+c)=dz;
                    });
                Precision sum_gz=Precision(0);
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team,width),
                    [&](const int c,Precision& sum) {
                        sum+=g(c)*scratch(width+c);
                    },sum_gz);
                Precision sum_gzy=Precision(0);
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team,width),
                    [&](const int c,Precision& sum) {
                        sum+=g(c)*scratch(width+c)*scratch(c);
                    },sum_gzy);
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team,width),
                    [&](const int c) {
                        const Precision y=scratch(c);
                        const Precision dz=scratch(width+c);
                        Kokkos::atomic_add(&h_adj_l(node,c),
                            inv*g(c)*dz-inv/Precision(width)*sum_gz
                            -inv*inv*inv*y/Precision(width)*sum_gzy);
                    });
                (void)a;
            });
    };
    // (2f) 稠密层反向(输入伴 = Σ_o W(o,i)·out̄(o);权重/偏置常量)。
    const auto dense_reverse=[&](Kokkos::View<const Precision**,Kokkos::LayoutRight> weight,
            Kokkos::View<const Precision**,Kokkos::LayoutRight> out_adj,
            Kokkos::View<Precision**,Kokkos::LayoutRight> in_adj,const bool initialize) {
        const int out_w=static_cast<int>(weight.extent(0));
        const int in_w=static_cast<int>(weight.extent(1));
        if(initialize) ordered_kokkos_deep_copy(in_adj,Precision(0));
        auto w=weight;auto o_adj=out_adj;auto i_adj=in_adj;
        // Parallelise over (node, input channel).  Each (node,i) slot has a
        // single writer and keeps the original o=0..out_w-1 accumulation
        // order, so the result is bitwise identical to the old per-node loop
        // while exposing num_nodes*in_w threads instead of num_nodes.
        Kokkos::parallel_for("polar reverse dense",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                {num_nodes,in_w}),
            KOKKOS_LAMBDA(int node,int i) {
                Precision acc=Precision(0);
                for(int o=0;o<out_w;++o)
                    acc+=w(o,i)*o_adj(node,o);
                i_adj(node,i)+=acc;
            });
    };
    // (2g) 共用: ρ̄ → assemble 反向(源伴 q̄ 写 dest 的 4 列;相位伴并入 r̄)。
    //      source(N,8) 提供 q = source(node,section_offset..+3)。
    const auto kspace_assemble_reverse=[&](Kokkos::View<const Precision*> rho_bar_re,
            Kokkos::View<const Precision*> rho_bar_im,
            Kokkos::View<const Precision**,Kokkos::LayoutRight> source,const int section_offset,
            Kokkos::View<Precision**,Kokkos::LayoutRight> dest,const int dest_offset) {
        auto dbre=polar_density_basis_re;auto dbim=polar_density_basis_im;
        ensure_view(polar_rev_acc_cos,polar_rev_acc_cos_storage,num_k,4);
        ensure_view(polar_rev_acc_sin,polar_rev_acc_sin_storage,num_k,4);
        auto acc_cos=polar_rev_acc_cos,acc_sin=polar_rev_acc_sin;
        auto kcos=polar_cosines,ksin=polar_sines;
        auto kv=polar_k_vectors,pos_adj=polar_rev_position_adjoint;
        const Precision rho_scale=Precision(two_pi_cubed/polar_cell_volume);
        Kokkos::parallel_for("polar rev assemble",num_k,
            KOKKOS_LAMBDA(int k) {
                for(int m=0;m<4;++m) {
                    acc_cos(k,m)=rho_scale*(rho_bar_re(k)*dbre(k,m)
                        +rho_bar_im(k)*dbim(k,m));
                    acc_sin(k,m)=rho_scale*(rho_bar_re(k)*dbim(k,m)
                        -rho_bar_im(k)*dbre(k,m));
                }
            });
        {
            const int source_chunks=4;
            const int source_chunk_k=
                (num_k+source_chunks-1)/source_chunks;
            ensure_view(polar_rev_phase_source_partial,
                polar_rev_phase_source_partial_storage,num_nodes,
                source_chunks*7);
            auto partial=polar_rev_phase_source_partial;
            Kokkos::parallel_for(
                "polar rev phase and source partial",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                    {0,0},{num_nodes,source_chunks}),
                KOKKOS_LAMBDA(int node,int chunk) {
                    const int k0=chunk*source_chunk_k;
                    const int k1=Kokkos::min(
                        num_k,k0+source_chunk_k);
                    Precision q_bar[4]={Precision(0),Precision(0),
                        Precision(0),Precision(0)};
                    Precision rx=Precision(0),ry=Precision(0),
                        rz=Precision(0);
                    for(int k=k0;k<k1;++k) {
                        const Precision c=kcos(k,node);
                        const Precision s=ksin(k,node);
                        for(int m=0;m<4;++m)
                            q_bar[m]+=c*acc_cos(k,m)
                                +s*acc_sin(k,m);
                        Precision cos_bar=Precision(0),
                            sin_bar=Precision(0);
                        for(int m=0;m<4;++m) {
                            const Precision q=
                                source(node,section_offset+m);
                            cos_bar+=acc_cos(k,m)*q;
                            sin_bar+=acc_sin(k,m)*q;
                        }
                        const Precision inner=-s*cos_bar+c*sin_bar;
                        rx+=kv(k,0)*inner;
                        ry+=kv(k,1)*inner;
                        rz+=kv(k,2)*inner;
                    }
                    for(int m=0;m<4;++m)
                        partial(node,7*chunk+m)=q_bar[m];
                    partial(node,7*chunk+4)=rx;
                    partial(node,7*chunk+5)=ry;
                    partial(node,7*chunk+6)=rz;
                });
            Kokkos::parallel_for("polar rev phase and source",
                num_nodes,KOKKOS_LAMBDA(int node) {
                    Precision q_bar[4]={Precision(0),Precision(0),
                        Precision(0),Precision(0)};
                    Precision rx=Precision(0),ry=Precision(0),
                        rz=Precision(0);
                    for(int chunk=0;chunk<source_chunks;++chunk) {
                        for(int m=0;m<4;++m)
                            q_bar[m]+=partial(node,7*chunk+m);
                        rx+=partial(node,7*chunk+4);
                        ry+=partial(node,7*chunk+5);
                        rz+=partial(node,7*chunk+6);
                    }
                    for(int m=0;m<4;++m)
                        Kokkos::atomic_add(&dest(node,dest_offset+m),
                            q_bar[m]);
                    pos_adj(node,0)+=rx;
                    pos_adj(node,1)+=ry;
                    pos_adj(node,2)+=rz;
                });
        }
    };
    // (2h) 描述符逐通道反向(feats̄(N,8) → q̄ + 相位伴;密度源=d_in)。
    const auto descriptor_reverse_channel=[&](const int channel,
            Kokkos::View<const Precision**,Kokkos::LayoutRight> density_in) {
        auto feats_bar=polar_rev_feats_adjoint;
        auto channel_adj=polar_rev_channel_adjoint;
        auto si_sel=polar_si_select;
        auto si_ov=polar_si_overlap;
        auto permutation=polar_output_permutation;
        auto fbre=polar_feature_basis_re,fbim=polar_feature_basis_im;
        auto kn2=polar_k_norm2,k0m=polar_k0_mask;
        auto kcos=polar_cosines,ksin=polar_sines;
        auto kweight=polar_k_weight;
        ensure_view(polar_rev_ab,polar_rev_ab_storage,num_k,16);
        ensure_view(polar_rev_rho_adj_re,polar_rev_rho_adj_re_storage,num_k);
        ensure_view(polar_rev_rho_adj_im,polar_rev_rho_adj_im_storage,num_k);
        auto ab=polar_rev_ab;
        auto rho_bar_re=polar_rev_rho_adj_re,rho_bar_im=polar_rev_rho_adj_im;
        const Precision proj_scale=Precision(2.0/two_pi_cubed);
        // (i) 自作用子项反(sourcē += selectᵀ(C ⊙ (−feats̄)))。
        Kokkos::parallel_for("polar desc si reverse",num_nodes,
            KOKKOS_LAMBDA(int node) {
                for(int j=0;j<8;++j)
                    Kokkos::atomic_add(&channel_adj(node,
                            4*channel+si_sel(j)),
                        -si_ov(j)*feats_bar(node,j));
            });
        // (ii) Ā/Ŷ = Σ_i cos/sin·proj_scale·feat̄(逆置换并入:j ↦ m=perm(j) 为双射)。
        // Round-3: split the node reduction into four contiguous chunks so
        // the kernel uses (num_k*8*4) threads instead of one block; the
        // chunks are folded in ascending order, so the result is
        // deterministic.
        {
            const int ab_chunks=4;
            const int ab_chunk_nodes=
                (owned_nodes+ab_chunks-1)/ab_chunks;
            ensure_view(polar_rev_ab_partial,
                polar_rev_ab_partial_storage,num_k,8*ab_chunks*2);
            auto ab_partial=polar_rev_ab_partial;
            Kokkos::parallel_for("polar desc ab reverse partial",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                    {0,0},{num_k,8*ab_chunks}),
                KOKKOS_LAMBDA(int k,int jc) {
                    const int j=jc/ab_chunks,chunk=jc%ab_chunks;
                    const int n0=chunk*ab_chunk_nodes;
                    const int n1=Kokkos::min(
                        owned_nodes,n0+ab_chunk_nodes);
                    const Precision w=kweight(k);
                    Precision a=Precision(0),b=Precision(0);
                    for(int node=n0;node<n1;++node) {
                        const Precision z=w*proj_scale
                            *feats_bar(node,j);
                        a+=kcos(k,node)*z;
                        b+=ksin(k,node)*z;
                    }
                    ab_partial(k,2*(j*ab_chunks+chunk))=a;
                    ab_partial(k,2*(j*ab_chunks+chunk)+1)=b;
                });
            Kokkos::parallel_for("polar desc ab reverse",num_k*8,
                KOKKOS_LAMBDA(int index) {
                    const int k=index/8,j=index%8;
                    Precision a=Precision(0),b=Precision(0);
                    for(int chunk=0;chunk<ab_chunks;++chunk) {
                        a+=ab_partial(k,2*(j*ab_chunks+chunk));
                        b+=ab_partial(k,2*(j*ab_chunks+chunk)+1);
                    }
                    const int m=permutation(j);
                    ab(k,m)=a;ab(k,8+m)=b;
                });
        }
        {
            Kokkos::View<Precision*> ab_flat(ab.data(),ab.size());
            polar_global_sum_view(ab_flat);
        }
        // (iii) project+kernel 反向一步到位:
        //       pr̄=Σ_m(X̄·FBR+Ŷ·FBI); pī=Σ_m(X̄·FBI−Ŷ·FBR); ρ̄=pr̄·inv_k2。
        Kokkos::parallel_for("polar desc rho adjoint",num_k,
            KOKKOS_LAMBDA(int k) {
                const Precision kfp=(k0m(k)>Precision(0))
                    ?Precision(0.5):Precision(1);
                const Precision inv_k2=(k0m(k)>Precision(0))
                    ?Precision(0):Precision(polar_field_constant)/kn2(k);
                Precision rr=Precision(0),ri=Precision(0);
                for(int m=0;m<8;++m) {
                    const Precision x_bar=ab(k,m)*kfp;
                    const Precision y_bar=ab(k,8+m)*kfp;
                    const Precision pr_bar=x_bar*fbre(k,m)+y_bar*fbim(k,m);
                    const Precision pi_bar=x_bar*fbim(k,m)-y_bar*fbre(k,m);
                    rr+=pr_bar*inv_k2;ri+=pi_bar*inv_k2;
                }
                rho_bar_re(k)=rr;rho_bar_im(k)=ri;
            });
        // (iv) assemble 反向 → q̄(4 列)+ 相位。
        kspace_assemble_reverse(rho_bar_re,rho_bar_im,density_in,
            4*channel,channel_adj,4*channel);
        // (iii-b) 项目相位伴随【v27 修复: 设计 §2 ⑧-3 (d3) 第二行原文此前漏落】:
        //   cos̄_proj(k,n) = Σ_m fa(k,m)·proj·feat̄(n,j(m));  sin̄_proj 同式(fb);
        //   pos += Σ_k k·[−s·cos̄_proj + c·sin̄_proj];
        //   fa/fb = 前向投影系数, 由本通道密度重算 ρ 后闭式:
        //   fa=(pr·FBR+pi·FBI)·kfp;  fb=(pr·FBI−pi·FBR)·kfp;  pr/pi=ρ·inv_k2。
        // (feat̄ 已经 (vi) /knorm 归一 ⇒ 与 (ii) 同口径。)
        {
            auto feats_bar=polar_rev_feats_adjoint;
            auto kv=polar_k_vectors;
            auto pos_adj=polar_rev_position_adjoint;
            ensure_view(polar_rev_rho_fwd_re,polar_rev_rho_fwd_re_storage,
                num_k);
            ensure_view(polar_rev_rho_fwd_im,polar_rev_rho_fwd_im_storage,
                num_k);
            auto rho_re=polar_rev_rho_fwd_re,rho_im=polar_rev_rho_fwd_im;
            { // ρ(本通道)重算(同 replay / ⑥ 同式同表)
                auto dbre=polar_density_basis_re,dbim=polar_density_basis_im;
                const Precision rho_scale=Precision(two_pi_cubed
                    /polar_cell_volume);
                Kokkos::parallel_for("polar desc proj rho recompute",num_k,
                    KOKKOS_LAMBDA(int k) {
                        Precision acc_cos[4],acc_sin[4];
                        for(int m=0;m<4;++m) {
                            acc_cos[m]=Precision(0);acc_sin[m]=Precision(0);
                        }
                        for(int node=0;node<owned_nodes;++node) {
                            const Precision c=kcos(k,node);
                            const Precision s=ksin(k,node);
                            for(int m=0;m<4;++m) {
                                const Precision q=density_in(node,4*channel+m);
                                acc_cos[m]+=c*q;acc_sin[m]+=s*q;
                            }
                        }
                        Precision re=Precision(0),im=Precision(0);
                        for(int m=0;m<4;++m) {
                            re+=dbre(k,m)*acc_cos[m]+dbim(k,m)*acc_sin[m];
                            im+=dbim(k,m)*acc_cos[m]-dbre(k,m)*acc_sin[m];
                        }
                        rho_re(k)=rho_scale*re;rho_im(k)=rho_scale*im;
                    });
                polar_global_sum_view(rho_re);
                polar_global_sum_view(rho_im);
            }
            {
                const int phase_chunks=4;
                const int phase_chunk_k=
                    (num_k+phase_chunks-1)/phase_chunks;
                ensure_view(polar_rev_phase_partial,
                    polar_rev_phase_partial_storage,num_nodes,
                    phase_chunks*3);
                auto phase_partial=polar_rev_phase_partial;
                Kokkos::parallel_for(
                    "polar desc proj phase reverse partial",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                        {0,0},{num_nodes,phase_chunks}),
                    KOKKOS_LAMBDA(int node,int chunk) {
                        const int k0=chunk*phase_chunk_k;
                        const int k1=Kokkos::min(
                            num_k,k0+phase_chunk_k);
                        Precision rx=Precision(0),ry=Precision(0),
                            rz=Precision(0);
                        for(int k=k0;k<k1;++k) {
                            const Precision kfp=(k0m(k)>Precision(0))
                                ?Precision(0.5):Precision(1);
                            const Precision inv_k2=(k0m(k)>Precision(0))
                                ?Precision(0)
                                :Precision(polar_field_constant)/kn2(k);
                            const Precision pr=rho_re(k)*inv_k2;
                            const Precision pi=rho_im(k)*inv_k2;
                            const Precision w=kweight(k);
                            Precision exp_c=Precision(0),
                                exp_s=Precision(0);
                            for(int j=0;j<8;++j) {
                                const int m=permutation(j);
                                const Precision z=w*proj_scale
                                    *feats_bar(node,j);
                                exp_c+=z*fbre(k,m);
                                exp_s+=z*fbim(k,m);
                            }
                            const Precision ex_c=
                                kfp*(pr*exp_c+pi*exp_s);
                            const Precision ex_s=
                                kfp*(pr*exp_s-pi*exp_c);
                            const Precision inner=-ksin(k,node)*ex_c
                                +kcos(k,node)*ex_s;
                            rx+=kv(k,0)*inner;
                            ry+=kv(k,1)*inner;
                            rz+=kv(k,2)*inner;
                        }
                        phase_partial(node,3*chunk)=rx;
                        phase_partial(node,3*chunk+1)=ry;
                        phase_partial(node,3*chunk+2)=rz;
                    });
                Kokkos::parallel_for("polar desc proj phase reverse",
                    num_nodes,KOKKOS_LAMBDA(int node) {
                        Precision rx=Precision(0),ry=Precision(0),
                            rz=Precision(0);
                        for(int chunk=0;chunk<phase_chunks;++chunk) {
                            rx+=phase_partial(node,3*chunk);
                            ry+=phase_partial(node,3*chunk+1);
                            rz+=phase_partial(node,3*chunk+2);
                        }
                        Kokkos::atomic_add(&pos_adj(node,0),rx);
                        Kokkos::atomic_add(&pos_adj(node,1),ry);
                        Kokkos::atomic_add(&pos_adj(node,2),rz);
                    });
            }
        }
    };
    // (2i) 逐步前向重放(③④ + ⑤; 输入 d_in,输出写 rev 视图;与⑤已落码逐算子镜像)。
    const auto replay_step_forward=[&](const int step,
            Kokkos::View<const Precision**,Kokkos::LayoutRight> density_in) {
        auto& update=polar_updates[step];
        ensure_view(polar_rev_step_potential,polar_rev_step_potential_storage,
            num_nodes,16);
        ensure_view(polar_rev_rho_fwd_re,polar_rev_rho_fwd_re_storage,num_k);
        ensure_view(polar_rev_rho_fwd_im,polar_rev_rho_fwd_im_storage,num_k);
        auto step_potential=polar_rev_step_potential;
        for(int channel=0;channel<2;++channel) {
            { // assemble(与⑤"polar scf assemble fourier"同式)
                auto rho_re=polar_rev_rho_fwd_re,rho_im=polar_rev_rho_fwd_im;
                auto dbre=polar_density_basis_re,dbim=polar_density_basis_im;
                auto kcos=polar_cosines,ksin=polar_sines;
                const Precision rho_scale=Precision(two_pi_cubed
                    /polar_cell_volume);
                Kokkos::parallel_for("polar rev replay assemble",num_k,
                    KOKKOS_LAMBDA(int k) {
                        Precision acc_cos[4],acc_sin[4];
                        for(int m=0;m<4;++m) {
                            acc_cos[m]=Precision(0);acc_sin[m]=Precision(0);
                        }
                        for(int node=0;node<owned_nodes;++node) {
                            const Precision c=kcos(k,node);
                            const Precision s=ksin(k,node);
                            for(int m=0;m<4;++m) {
                                const Precision q=density_in(node,4*channel+m);
                                acc_cos[m]+=c*q;acc_sin[m]+=s*q;
                            }
                        }
                        Precision re=Precision(0),im=Precision(0);
                        for(int m=0;m<4;++m) {
                            re+=dbre(k,m)*acc_cos[m]+dbim(k,m)*acc_sin[m];
                            im+=dbim(k,m)*acc_cos[m]-dbre(k,m)*acc_sin[m];
                        }
                        rho_re(k)=rho_scale*re;rho_im(k)=rho_scale*im;
                    });
                polar_global_sum_view(rho_re);
                polar_global_sum_view(rho_im);
            }
            { // project + SI + permute + /knorm(与⑤"polar scf project features"
              //  及随后的归一 kernel 逐元素等价;仅省去中间 feats 存储)
                auto rho_re=polar_rev_rho_fwd_re,rho_im=polar_rev_rho_fwd_im;
                auto fbre=polar_feature_basis_re,fbim=polar_feature_basis_im;
                auto kn2=polar_k_norm2,k0m=polar_k0_mask;
                auto kcos=polar_cosines,ksin=polar_sines;
                auto si_ov=polar_si_overlap;
                auto si_sel=polar_si_select;
                auto permutation=polar_output_permutation;
                auto knorm=polar_field_feature_norms;
                auto kweight=polar_k_weight;
                const Precision proj_scale=Precision(2.0/two_pi_cubed);
                Kokkos::parallel_for("polar rev replay project",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                        {num_nodes,8}),
                    KOKKOS_LAMBDA(int node,int j) {
                        const int m=permutation(j);
                        Precision proj_cos=Precision(0);
                        Precision proj_sin=Precision(0);
                        for(int k=0;k<num_k;++k) {
                            const Precision inv_k2=(k0m(k)>Precision(0))
                                ?Precision(0)
                                :Precision(polar_field_constant)/kn2(k);
                            const Precision pr=rho_re(k)*inv_k2;
                            const Precision pi=rho_im(k)*inv_k2;
                            const Precision kfp=(k0m(k)>Precision(0))
                                ?Precision(0.5):Precision(1);
                            const Precision fa=(pr*fbre(k,m)
                                +pi*fbim(k,m))*kfp;
                            const Precision fb=(pr*fbim(k,m)
                                -pi*fbre(k,m))*kfp;
                            proj_cos+=(kweight(k)*fa)*kcos(k,node);
                            proj_sin+=(kweight(k)*fb)*ksin(k,node);
                        }
                        const Precision value=proj_scale
                            *(proj_cos+proj_sin)
                            -si_ov(j)*density_in(node,
                                4*channel+si_sel(j));
                        step_potential(node,8*channel+j)=value/knorm(j);
                    });
            }
        }
        const int dp_width=update.source_embedding.output_dimension();
        const int hidden=update.nl_hidden_width;
        const int gate_w=update.gate_scalar_count+3*update.gate_gated_count;
        const int ro_w=update.readout_linear_1.output_dimension();
        ensure_view(polar_rev_pe,polar_rev_pe_storage,num_nodes,
            update.potential_linear.output_dimension());
        ensure_view(polar_rev_dp,polar_rev_dp_storage,num_nodes,dp_width);
        ensure_view(polar_rev_se,polar_rev_se_storage,num_nodes,dp_width);
        ensure_view(polar_rev_nl_in,polar_rev_nl_in_storage,num_nodes,
            2*dp_width);
        ensure_view(polar_rev_nl_h,polar_rev_nl_h_storage,num_nodes,3*hidden);
        ensure_view(polar_rev_nl_a,polar_rev_nl_a_storage,num_nodes,3*hidden);
        ensure_view(polar_rev_nl_out,polar_rev_nl_out_storage,num_nodes,
            static_cast<int>(update.nl_w3.extent(0)));
        ensure_view(polar_rev_tp,polar_rev_tp_storage,num_nodes,
            static_cast<int>(update.readout_linear_1.input_dimension()));
        ensure_view(polar_rev_ro1,polar_rev_ro1_storage,num_nodes,ro_w);
        ensure_view(polar_rev_gate1,polar_rev_gate1_storage,num_nodes,gate_w);
        ensure_view(polar_rev_ro2,polar_rev_ro2_storage,num_nodes,ro_w);
        ensure_view(polar_rev_gate2,polar_rev_gate2_storage,num_nodes,gate_w);
        ensure_view(polar_rev_out,polar_rev_out_storage,num_nodes,
            static_cast<int>(update.readout_linear_2.output_dimension()));
        { // 势嵌入三分量(与⑤同序:potential_linear + node_feats_linear + charge_embedding)
            update.potential_linear.evaluate(polar_rev_step_potential,
                polar_rev_pe);
            update.node_feats_linear.evaluate(polar_features_mixed,
                polar_scratch_a);
            update.charge_embedding.evaluate(density_in,polar_scratch_b);
            auto pe=polar_rev_pe,ra=polar_scratch_a,rb=polar_scratch_b;
            Kokkos::parallel_for("polar rev replay pe sum",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                    {num_nodes,static_cast<int>(pe.extent(1))}),
                KOKKOS_LAMBDA(int node,int column) {
                    pe(node,column)+=ra(node,column)+rb(node,column);
                });
        }
        const auto to_host_vector=[&](const auto& view) {
            const auto mirror=Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(),view);
            std::vector<typename decltype(mirror)::value_type>
                values(mirror.size());
            for(std::size_t i2=0;i2<values.size();++i2)
                values[i2]=mirror.data()[i2];
            return values;
        };
        const std::vector<int> dot_meta=to_host_vector(update.uvu_dot_meta);
        const std::vector<Precision> dot_pw=
            to_host_vector(update.uvu_dot_path_weight);
        const std::vector<int> tp_meta=to_host_vector(update.uvu_tp_meta);
        const std::vector<Precision> tp_pw=
            to_host_vector(update.uvu_tp_path_weight);
        const auto launch_uvu_forward=[&](const std::vector<int>& meta,
                const std::vector<Precision>& path_weight,
                Kokkos::View<Precision*> weight_view,
                Kokkos::View<Precision*> mask_view,
                Kokkos::View<const Precision**,Kokkos::LayoutRight> x1,
                Kokkos::View<const Precision**,Kokkos::LayoutRight> x2,
                Kokkos::View<Precision**,Kokkos::LayoutRight> output) {
            const int paths=static_cast<int>(meta.size())/8;
            const int width=static_cast<int>(output.extent(1));
            Kokkos::parallel_for("polar rev replay uvu zero",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                    {num_nodes,width}),
                KOKKOS_LAMBDA(int node,int column) {
                    output(node,column)=Precision(0);
                });
            for(int p=0;p<paths;++p) {
                const int in1_off=meta[p*8+0],in2_off=meta[p*8+1],
                    out_off=meta[p*8+2],w_off=meta[p*8+3],
                    mul1=meta[p*8+4],mul2=meta[p*8+5],mode=meta[p*8+6],
                    d1=meta[p*8+7];
                const Precision pw=path_weight[p];
                const Precision inv=Precision(1)/Kokkos::sqrt(Precision(d1));
                if(mode==0) {
                    if(d1<=8) {
                        const int tile_n=64,tile_m=4;
                        const int n_tiles=(num_nodes+tile_n-1)/tile_n;
                        const int m_tiles=(mul1+tile_m-1)/tile_m;
                        using scratch_view=Kokkos::View<Precision*,
                            Kokkos::DefaultExecutionSpace::
                                scratch_memory_space,
                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
                        const int stride2=mul2+1;
                        const std::size_t smem=scratch_view::shmem_size(
                            tile_m*stride2);
                        Kokkos::parallel_for("polar rev replay uvu mode0",
                            Kokkos::TeamPolicy<>(
                                n_tiles*m_tiles,tile_n*tile_m)
                                .set_scratch_size(0,Kokkos::PerTeam(smem)),
                            KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::
                                member_type& team) {
                                const int tile=team.league_rank();
                                const int nt=tile/m_tiles,ut=tile%m_tiles;
                                const int n0=nt*tile_n,u0=ut*tile_m;
                                scratch_view s(team.team_scratch(0),
                                    tile_m*stride2);
                                for(int idx=team.team_rank();
                                    idx<tile_m*mul2;
                                    idx+=team.team_size()) {
                                    const int uu=idx/mul2,vv=idx%mul2;
                                    const int u=u0+uu;
                                    s(uu*stride2+vv)=(u<mul1)
                                        ?weight_view(w_off+u*mul2+vv)
                                        :Precision(0);
                                }
                                team.team_barrier();
                                const int tr=team.team_rank();
                                const int node=n0+tr/tile_m,uu=tr%tile_m;
                                if(node<num_nodes&&u0+uu<mul1) {
                                    const int u=u0+uu;
                                    Precision x1v[8];
                                    for(int c=0;c<d1;++c)
                                        x1v[c]=x1(node,in1_off+u*d1+c);
                                    Precision mixed=Precision(0);
                                    for(int v=0;v<mul2;++v) {
                                        Precision pair=Precision(0);
                                        for(int c=0;c<d1;++c)
                                            pair+=x1v[c]
                                                *x2(node,in2_off+v*d1+c);
                                        mixed+=pair*inv*s(uu*stride2+v);
                                    }
                                    output(node,out_off+u)+=pw*mixed;
                                }
                            });
                    } else {
                    Kokkos::parallel_for("polar rev replay uvu mode0",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                            {0,0},{num_nodes,mul1}),
                        KOKKOS_LAMBDA(int node,int u) {
                            Precision mixed=Precision(0);
                            for(int v=0;v<mul2;++v) {
                                Precision pair=Precision(0);
                                for(int c=0;c<d1;++c)
                                    pair+=x1(node,in1_off+u*d1+c)
                                        *x2(node,in2_off+v*d1+c);
                                mixed+=pair*inv*weight_view(w_off+u*mul2+v);
                            }
                            output(node,out_off+u)+=pw*mixed;
                        });
                    }
                } else {
                    Kokkos::parallel_for("polar rev replay uvu mode1",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                            {0,0},{num_nodes,mul1}),
                        KOKKOS_LAMBDA(int node,int u) {
                            Precision mixed=Precision(0);
                            for(int v=0;v<mul2;++v)
                                mixed+=x2(node,in2_off+v)
                                    *weight_view(w_off+u*mul2+v);
                            const Precision contribution=pw*mixed*inv;
                            for(int c=0;c<d1;++c)
                                output(node,out_off+u*d1+c)+=contribution
                                    *x1(node,in1_off+u*d1+c);
                        });
                }
            }
            Kokkos::parallel_for("polar rev replay uvu mask",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                    {num_nodes,width}),
                KOKKOS_LAMBDA(int node,int column) {
                    output(node,column)*=mask_view(column);
                });
        };
        launch_uvu_forward(dot_meta,dot_pw,update.uvu_dot_weight,
            update.uvu_dot_mask,polar_features_mixed,polar_rev_pe,
            polar_rev_dp);
        update.source_embedding.evaluate(attrs,polar_rev_se);
        { // nl_in = cat(dp, se)
            auto ni=polar_rev_nl_in,dp=polar_rev_dp,se=polar_rev_se;
            Kokkos::parallel_for("polar rev replay nl concat",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                    {num_nodes,dp_width}),
                KOKKOS_LAMBDA(int node,int column) {
                    ni(node,column)=dp(node,column);
                    ni(node,dp_width+column)=se(node,column);
                });
        }
        { // MLP 重放: h_k → nl_h 块 k;a_k → nl_a 块 k(k=0..2);dense3 → nl_out
            auto h=polar_rev_nl_h;auto a=polar_rev_nl_a;
            auto ni=polar_rev_nl_in;auto out_nl=polar_rev_nl_out;
            const Precision eps=update.ln_eps;
            const auto ln_block=[&](Kokkos::View<Precision**,Kokkos::LayoutRight> hb,
                    Kokkos::View<Precision**,Kokkos::LayoutRight> ab,Kokkos::View<const Precision*> g,
                    Kokkos::View<const Precision*> b) {
                const int ln_team=hidden<128?hidden:128;
                Kokkos::parallel_for("polar rev replay ln",
                    Kokkos::TeamPolicy<>(num_nodes,ln_team),
                    KOKKOS_LAMBDA(
                        const Kokkos::TeamPolicy<>::member_type& team) {
                        const int node=team.league_rank();
                        Precision mean=Precision(0);
                        Kokkos::parallel_reduce(
                            Kokkos::TeamThreadRange(team,hidden),
                            [&](const int c2,Precision& sum) {
                                sum+=hb(node,c2);
                            },mean);
                        mean/=Precision(hidden);
                        Precision var=Precision(0);
                        Kokkos::parallel_reduce(
                            Kokkos::TeamThreadRange(team,hidden),
                            [&](const int c2,Precision& sum) {
                                const Precision delta=hb(node,c2)-mean;
                                sum+=delta*delta;
                            },var);
                        var/=Precision(hidden);
                        const Precision inv=Precision(1)
                            /Kokkos::sqrt(var+eps);
                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team,hidden),
                            [&](const int c2) {
                                const Precision normalized=(hb(node,c2)
                                    -mean)*inv*g(c2)+b(c2);
                                ab(node,c2)=normalized
                                    /(Precision(1)+Kokkos::exp(
                                        -normalized));
                            });
                    });
            };
            const auto dense_forward=[&](Kokkos::View<const Precision**,Kokkos::LayoutRight> w,
                    Kokkos::View<const Precision*> b,
                    Kokkos::View<const Precision**,Kokkos::LayoutRight> src,
                    Kokkos::View<Precision**,Kokkos::LayoutRight> dst,int dst_offset,
                    const int src_width) {
                Kokkos::parallel_for("polar rev replay dense",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                        {num_nodes,static_cast<int>(w.extent(0))}),
                    KOKKOS_LAMBDA(int node,int o) {
                        Precision acc=b(o);
                        for(int c2=0;c2<src_width;++c2)
                            acc+=w(o,c2)*src(node,c2);
                        dst(node,dst_offset+o)=acc;
                    });
            };
            {
                auto w0=update.nl_w0;auto b0=update.nl_b0;
                const int src_width=static_cast<int>(w0.extent(1));
                Kokkos::parallel_for("polar rev replay dense0",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                        {num_nodes,static_cast<int>(w0.extent(0))}),
                    KOKKOS_LAMBDA(int node,int o) {
                        Precision acc=b0(o);
                        for(int c2=0;c2<src_width;++c2)
                            acc+=w0(o,c2)*ni(node,c2);
                        h(node,o)=acc;
                    });
            }
            ln_block(Kokkos::subview(h,Kokkos::ALL,
                    std::make_pair(0,hidden)),
                Kokkos::subview(a,Kokkos::ALL,std::make_pair(0,hidden)),
                update.ln0_weight,update.ln0_bias);
            dense_forward(update.nl_w1,update.nl_b1,
                Kokkos::subview(a,Kokkos::ALL,std::make_pair(0,hidden)),
                h,hidden,hidden);
            ln_block(Kokkos::subview(h,Kokkos::ALL,
                    std::make_pair(hidden,2*hidden)),
                Kokkos::subview(a,Kokkos::ALL,
                    std::make_pair(hidden,2*hidden)),
                update.ln1_weight,update.ln1_bias);
            dense_forward(update.nl_w2,update.nl_b2,
                Kokkos::subview(a,Kokkos::ALL,
                    std::make_pair(hidden,2*hidden)),
                h,2*hidden,hidden);
            ln_block(Kokkos::subview(h,Kokkos::ALL,
                    std::make_pair(2*hidden,3*hidden)),
                Kokkos::subview(a,Kokkos::ALL,
                    std::make_pair(2*hidden,3*hidden)),
                update.ln2_weight,update.ln2_bias);
            { // dense3 → nl_out(末层无 LN)
                auto w3=update.nl_w3;auto b3=update.nl_b3;
                const int src_width=static_cast<int>(w3.extent(1));
                Kokkos::parallel_for("polar rev replay dense3",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                        {num_nodes,static_cast<int>(w3.extent(0))}),
                    KOKKOS_LAMBDA(int node,int o) {
                        Precision acc=b3(o);
                        for(int c2=0;c2<src_width;++c2)
                            acc+=w3(o,c2)*a(node,2*hidden+c2);
                        out_nl(node,o)=acc;
                    });
            }
        }
        launch_uvu_forward(tp_meta,tp_pw,update.uvu_tp_weight,
            update.uvu_tp_mask,polar_features_mixed,polar_rev_nl_out,
            polar_rev_tp);
        update.readout_linear_1.evaluate(polar_rev_tp,polar_rev_ro1);
        {
            const Precision scalar_cst=update.gate_scalar_cst;
            const Precision gate_cst=update.gate_gate_cst;
            const int scalars=update.gate_scalar_count;
            const int gates=update.gate_gate_count;
            auto src=polar_rev_ro1;auto dst=polar_rev_gate1;
            Kokkos::parallel_for("polar rev replay gate1",num_nodes,
                KOKKOS_LAMBDA(int node) {
                    for(int i2=0;i2<scalars;++i2) {
                        const Precision s=src(node,i2);
                        dst(node,i2)=scalar_cst*s
                            /(Precision(1)+Kokkos::exp(-s));
                    }
                    for(int i2=0;i2<gates;++i2) {
                        const Precision g=gate_cst
                            /(Precision(1)+Kokkos::exp(-src(node,scalars+i2)));
                        for(int c2=0;c2<3;++c2)
                            dst(node,scalars+3*i2+c2)=g
                                *src(node,scalars+gates+3*i2+c2);
                    }
                });
        }
        update.readout_linear_mid.evaluate(polar_rev_gate1,polar_rev_ro2);
        {
            const Precision scalar_cst=update.gate_scalar_cst;
            const Precision gate_cst=update.gate_gate_cst;
            const int scalars=update.gate_scalar_count;
            const int gates=update.gate_gate_count;
            auto src=polar_rev_ro2;auto dst=polar_rev_gate2;
            Kokkos::parallel_for("polar rev replay gate2",num_nodes,
                KOKKOS_LAMBDA(int node) {
                    for(int i2=0;i2<scalars;++i2) {
                        const Precision s=src(node,i2);
                        dst(node,i2)=scalar_cst*s
                            /(Precision(1)+Kokkos::exp(-s));
                    }
                    for(int i2=0;i2<gates;++i2) {
                        const Precision g=gate_cst
                            /(Precision(1)+Kokkos::exp(-src(node,scalars+i2)));
                        for(int c2=0;c2<3;++c2)
                            dst(node,scalars+3*i2+c2)=g
                                *src(node,scalars+gates+3*i2+c2);
                    }
                });
        }
        update.readout_linear_2.evaluate(polar_rev_gate2,polar_rev_out);
    };
    // (2i-b) Plan B1: load the forward per-step state cached by
    // run_polar_pipeline (MATOS_REPLAY_CACHE=1) instead of replaying ③④⑤.
    // Only the tensors actually consumed by the reverse are copied: pe is the
    // uvu_dot x2 operand, nl_out the uvu_tp x2 operand, nl_h/nl_a the MLP
    // activation inputs, ro1/ro2 the gate inputs, out the Fukui source.
    // dp/se/gate1/gate2/tp are never read by the reverse; the latter two are
    // ensured only for their extents (readout_linear_1 / nl_in adjoints).
    const auto load_step_cache=[&](const int step) {
        auto& update=polar_updates[step];
        const int dp_width=update.source_embedding.output_dimension();
        const int hidden=update.nl_hidden_width;
        const int ro_w=update.readout_linear_1.output_dimension();
        ensure_view(polar_rev_pe,polar_rev_pe_storage,num_nodes,
            update.potential_linear.output_dimension());
        ensure_view(polar_rev_nl_h,polar_rev_nl_h_storage,num_nodes,
            3*hidden);
        ensure_view(polar_rev_nl_a,polar_rev_nl_a_storage,num_nodes,
            3*hidden);
        ensure_view(polar_rev_nl_out,polar_rev_nl_out_storage,num_nodes,
            static_cast<int>(update.nl_w3.extent(0)));
        ensure_view(polar_rev_ro1,polar_rev_ro1_storage,num_nodes,ro_w);
        ensure_view(polar_rev_ro2,polar_rev_ro2_storage,num_nodes,ro_w);
        ensure_view(polar_rev_out,polar_rev_out_storage,num_nodes,
            static_cast<int>(update.readout_linear_2.output_dimension()));
        // Extent-only workspaces (contents unused by the reverse).
        ensure_view(polar_rev_tp,polar_rev_tp_storage,num_nodes,
            static_cast<int>(update.readout_linear_1.input_dimension()));
        ensure_view(polar_rev_nl_in,polar_rev_nl_in_storage,num_nodes,
            2*dp_width);
        Kokkos::deep_copy(polar_rev_pe,Kokkos::subview(
            polar_rev_cache_pe,step,Kokkos::ALL,Kokkos::ALL));
        Kokkos::deep_copy(polar_rev_nl_h,Kokkos::subview(
            polar_rev_cache_nl_h,step,Kokkos::ALL,Kokkos::ALL));
        Kokkos::deep_copy(polar_rev_nl_a,Kokkos::subview(
            polar_rev_cache_nl_a,step,Kokkos::ALL,Kokkos::ALL));
        Kokkos::deep_copy(polar_rev_nl_out,Kokkos::subview(
            polar_rev_cache_nl_out,step,Kokkos::ALL,Kokkos::ALL));
        Kokkos::deep_copy(polar_rev_ro1,Kokkos::subview(
            polar_rev_cache_ro1,step,Kokkos::ALL,Kokkos::ALL));
        Kokkos::deep_copy(polar_rev_ro2,Kokkos::subview(
            polar_rev_cache_ro2,step,Kokkos::ALL,Kokkos::ALL));
        Kokkos::deep_copy(polar_rev_out,Kokkos::subview(
            polar_rev_cache_out,step,Kokkos::ALL,Kokkos::ALL));
    };

        symmetrix_polar_profile_mark("rev1.5:le-seed");
// ---------- 1.5) [⑧-7] le 反向种子生产 ----------
    // 投递契约 module8_patch_final.md §4.1;语义 POLAR_REVERSE_DESIGN.md §2(7)/§3.2。
    // 位置 = readout 播种之后、种子判定之前(§4.0-A 接缝);同一 eval ≤1 次。
    // 前向锚(本文件 ⑦ 段): mlp_out=MLP(invar);invar=[dot_q|dot_v];
    //   dot_q/dot_v=uvu(x1=nf_last,x2=q_up/v_up);q_up=Lin_q(q_in);v_up=Lin_v(P_last);
    //   q_in=d_final+d_init(d_init=polar_field_independent_density) ⇒ 双分支同值。
    if(polar_le_enabled&&polar_le_add) {
        // [dbg 2026-10-16] 通道隔离开关(诊断用): bit0=跳过 d 种子, bit1=跳过 P 种子,
        // bit2=跳过 nf 注入。默认 0(全投递);对账后随探针一并清理。
        const char* dbg_skip_env=std::getenv("POLAR_DBG_SKIP");
        const int dbg_skip=dbg_skip_env?std::atoi(dbg_skip_env):0;
        if(static_cast<int>(polar_le_invar.extent(0))!=num_nodes
            ||static_cast<int>(polar_le_q_up.extent(0))!=num_nodes
            ||static_cast<int>(polar_le_v_up.extent(0))!=num_nodes
            ||static_cast<int>(polar_node_feats_e3nn.extent(0))!=num_nodes)
            throw std::runtime_error(
                "polar le reverse: forward buffers are not resident "
                "(local_electron_energy did not run this evaluation?).");
        const int le_dot_width=static_cast<int>(polar_le_q_mask.extent(0));
        const int le_up_width=static_cast<int>(polar_le_q_up.extent(1));
        // (1) MLP 反向: mlp_out̄≡1(e0 装配含 le 不加权) → invar̄(内部重算+带式)。
        ensure_view(polar_rev_le_mlp_out_bar,
            polar_rev_le_mlp_out_bar_storage,num_nodes,1);
        auto le_mlp_out_bar=polar_rev_le_mlp_out_bar;
        ensure_view(polar_rev_le_invar_bar,
            polar_rev_le_invar_bar_storage,num_nodes,2*le_dot_width);
        auto le_invar_bar=polar_rev_le_invar_bar;
        {
            auto bar=le_mlp_out_bar;
            Kokkos::parallel_for("polar le mlp_out adjoint init",num_nodes,
                KOKKOS_LAMBDA(int node) {
                    bar(node,0)=Precision(1);
                });
        }
        polar_le_mlp.reverse(polar_le_invar,le_mlp_out_bar,le_invar_bar);
        symmetrix_polar_profile_mark("rev1.5a:mlp-reverse");
        // (2) invar̄ 拆两路。
        ensure_view(polar_rev_le_dot_q_bar,
            polar_rev_le_dot_q_bar_storage,num_nodes,le_dot_width);
        auto le_dot_q_bar=polar_rev_le_dot_q_bar;
        ensure_view(polar_rev_le_dot_v_bar,
            polar_rev_le_dot_v_bar_storage,num_nodes,le_dot_width);
        auto le_dot_v_bar=polar_rev_le_dot_v_bar;
        {
            auto qb=le_dot_q_bar;
            auto vb=le_dot_v_bar;
            auto ib=le_invar_bar;
            Kokkos::parallel_for("polar le dot adjoint split",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                    {num_nodes,le_dot_width}),
                KOKKOS_LAMBDA(int node,int column) {
                    qb(node,column)=ib(node,column);
                    vb(node,column)=ib(node,le_dot_width+column);
                });
        }
        // (3) uvu 反向(复用 uvu_reverse;x1=nf_last ⇒ 两路 n̄f 累加)。
        ensure_view(polar_rev_le_q_up_bar,
            polar_rev_le_q_up_bar_storage,num_nodes,le_up_width);
        auto le_q_up_bar=polar_rev_le_q_up_bar;
        ensure_view(polar_rev_le_v_up_bar,
            polar_rev_le_v_up_bar_storage,num_nodes,le_up_width);
        auto le_v_up_bar=polar_rev_le_v_up_bar;
        ensure_view(polar_rev_le_nf_bar,
            polar_rev_le_nf_bar_storage,num_nodes,polar_feature_width);
        auto le_nf_bar=polar_rev_le_nf_bar;
        {
            const auto le_to_host=[&](const auto& view) {
                const auto mirror=Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(),view);
                std::vector<typename decltype(mirror)::value_type>
                    values(mirror.size());
                for(std::size_t i2=0;i2<values.size();++i2)
                    values[i2]=mirror.data()[i2];
                return values;
            };
            const std::vector<int> le_q_meta=le_to_host(polar_le_q_meta);
            const std::vector<Precision> le_q_pw=
                le_to_host(polar_le_q_path_weight);
            const std::vector<int> le_v_meta=le_to_host(polar_le_v_meta);
            const std::vector<Precision> le_v_pw=
                le_to_host(polar_le_v_path_weight);
            auto zero_fill=[&](const char* name,
                    Kokkos::View<Precision**,Kokkos::LayoutRight> v) {
                const int columns=static_cast<int>(v.extent(1));
                Kokkos::parallel_for(name,
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                        {num_nodes,columns}),
                    KOKKOS_LAMBDA(int node,int column) {
                        v(node,column)=Precision(0);
                    });
            };
            zero_fill("polar le q_up adjoint zero",le_q_up_bar);
            zero_fill("polar le v_up adjoint zero",le_v_up_bar);
            zero_fill("polar le nf adjoint zero",le_nf_bar);
            const auto le_nf_source=polar_node_feats_e3nn;
            const auto le_q_up_view=polar_le_q_up;
            const auto le_v_up_view=polar_le_v_up;
            uvu_reverse(le_q_meta,le_q_pw,polar_le_q_weight,
                polar_le_q_mask,le_nf_source,le_q_up_view,
                le_dot_q_bar,le_nf_bar,le_q_up_bar);
            uvu_reverse(le_v_meta,le_v_pw,polar_le_v_weight,
                polar_le_v_mask,le_nf_source,le_v_up_view,
                le_dot_v_bar,le_nf_bar,le_v_up_bar);
        }
        symmetrix_polar_profile_mark("rev1.5b:uvu-reverse");
        // (4) Lin_q/Lin_v 反向(仿射;权重内建)。
        ensure_view(polar_rev_le_q_in_bar,
            polar_rev_le_q_in_bar_storage,num_nodes,8);
        auto le_q_in_bar=polar_rev_le_q_in_bar;
        ensure_view(polar_rev_le_p_last_bar,
            polar_rev_le_p_last_bar_storage,num_nodes,16);
        auto le_p_last_bar=polar_rev_le_p_last_bar;
        polar_le_linear_up_q.reverse(le_q_up_bar,le_q_in_bar,
            /*initialize_input_adjoint=*/true);
        polar_le_linear_up_v.reverse(le_v_up_bar,le_p_last_bar,
            /*initialize_input_adjoint=*/true);
        symmetrix_polar_profile_mark("rev1.5c:linear-reverse");
        // [dbg] 导出伴随镜像(逐级对账用;只读,不参与链)。
        ensure_view(polar_dbg_le_qin_bar,polar_dbg_le_qin_bar_storage,
            num_nodes,8);
        ensure_view(polar_dbg_le_plast_bar,polar_dbg_le_plast_bar_storage,
            num_nodes,16);
        ensure_view(polar_dbg_le_nf_bar,polar_dbg_le_nf_bar_storage,
            num_nodes,polar_feature_width);
        ordered_kokkos_deep_copy(polar_dbg_le_qin_bar,le_q_in_bar);
        ordered_kokkos_deep_copy(polar_dbg_le_plast_bar,le_p_last_bar);
        ordered_kokkos_deep_copy(polar_dbg_le_nf_bar,le_nf_bar);
        // (5) 种子投递(§4.1;ensure+清零后累加;q_in=d_final+d_init 双分支同值)。
        ensure_view(polar_seed_density_adjoint,
            polar_seed_density_adjoint_storage,num_nodes,8);
        ensure_view(polar_seed_density0_adjoint,
            polar_seed_density0_adjoint_storage,num_nodes,8);
        ensure_view(polar_seed_potential_adjoint,
            polar_seed_potential_adjoint_storage,num_nodes,16);
        ordered_kokkos_deep_copy(polar_seed_density_adjoint,Precision(0));
        ordered_kokkos_deep_copy(polar_seed_density0_adjoint,Precision(0));
        ordered_kokkos_deep_copy(polar_seed_potential_adjoint,Precision(0));
        {
            auto sd=polar_seed_density_adjoint;
            auto sd0=polar_seed_density0_adjoint;
            auto sp=polar_seed_potential_adjoint;
            auto qb=le_q_in_bar;
            auto pb=le_p_last_bar;
            if(!(dbg_skip&1))
                Kokkos::parallel_for("polar le seeds density write",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{num_nodes,8}),
                    KOKKOS_LAMBDA(int node,int column) {
                        sd(node,column)+=qb(node,column);
                        sd0(node,column)+=qb(node,column);
                    });
            if(!(dbg_skip&2))
                Kokkos::parallel_for("polar le seeds potential write",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{num_nodes,16}),
                    KOKKOS_LAMBDA(int node,int column) {
                        sp(node,column)+=pb(node,column);
                    });
        }
        polar_reverse_seed_terms|=1;
        // (6) n̄f_last → ⑧-A 注入末层(工具为 atomic_add=init=false 语义)。
        if(!(dbg_skip&4))
            polar_inject_node_features_adjoint(
                static_cast<int>(states.size())-1,le_nf_bar);
        symmetrix_polar_profile_mark("rev1.5d:inject");
    }

        symmetrix_polar_profile_mark("rev2:early-exit");
// ---------- 2) 种子早退判定(掩码=0 ⇒ 零力返回,对①-⑤门零影响) ----------
    const int seed_terms=polar_reverse_seed_terms
        |(polar_coulomb_enabled?2:0);
    polar_reverse_terms_last=seed_terms;   // 诊断(非 consume-once;门脚本分档用)
    if(seed_terms==0) {
        polar_reverse_seed_terms=0;
        return;
    }

        symmetrix_polar_profile_mark("rev3:le-merge");
// ---------- 3) ⑦ le 种子并入(若落码): d_final / P_last 链头;d_init 留到 SCF 之后 ----------
    if(seed_terms&1) {
        if(static_cast<int>(polar_seed_density_adjoint.extent(0))!=num_nodes
            ||static_cast<int>(polar_seed_density_adjoint.extent(1))!=8
            ||static_cast<int>(polar_seed_density0_adjoint.extent(0))!=num_nodes
            ||static_cast<int>(polar_seed_density0_adjoint.extent(1))!=8
            ||static_cast<int>(polar_seed_potential_adjoint.extent(0))!=num_nodes
            ||static_cast<int>(polar_seed_potential_adjoint.extent(1))!=16)
            throw std::runtime_error(
                "polar reverse: le seed extents are inconsistent "
                "(see module8_patch_final.md 4.1).");
        auto dbar=polar_rev_density_adjoint;
        auto seed_final=polar_seed_density_adjoint;
        Kokkos::parallel_for("polar reverse seed density final",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{num_nodes,8}),
            KOKKOS_LAMBDA(int node,int column) {
                dbar(node,column)+=seed_final(node,column);
            });
    }

        symmetrix_polar_profile_mark("rev4:coulomb-seed");
// ---------- 4) ⑥ coulomb 种子(polar_coulomb_enabled 由 ⑥ 落码置位) ----------
    // E = 0.5·V/(2π)⁶·Σ_k 2(ρ_r·pot_r+ρ_i·pot_i),pot=FIELD_CONSTANT·kfac_c·ρ ⇒
    // ρ̄ = 2·V/(2π)⁶·FIELD_CONSTANT·kfac_c·ρ;q(n,m)=d_final(n,m)+d_final(n,4+m)。
    if(seed_terms&2) {
        ensure_view(polar_rev_source_adjoint,polar_rev_source_adjoint_storage,
            num_nodes,8);
        ensure_view(polar_rev_channel_adjoint,polar_rev_channel_adjoint_storage,
            num_nodes,8);
        ordered_kokkos_deep_copy(polar_rev_channel_adjoint,Precision(0));
        {
            auto q_view=polar_rev_source_adjoint;
            // [v3 适配·D-05] ⑥ 前向密度源 = polar_scf_final_density(polar TU ⑥-C1 实读);
            // polar_spin_charge_density 自 v19c 起为只读①原始累积, 与 SCF 末态不等价。
            auto final_density=polar_scf_final_density;
            Kokkos::parallel_for("polar coulomb q",num_nodes,
                KOKKOS_LAMBDA(int node) {
                    for(int m=0;m<4;++m)
                        q_view(node,m)=final_density(node,m)
                            +final_density(node,4+m);
                });
        }
        ensure_view(polar_rev_rho_fwd_re,polar_rev_rho_fwd_re_storage,num_k);
        ensure_view(polar_rev_rho_fwd_im,polar_rev_rho_fwd_im_storage,num_k);
        ensure_view(polar_rev_rho_adj_re,polar_rev_rho_adj_re_storage,num_k);
        ensure_view(polar_rev_rho_adj_im,polar_rev_rho_adj_im_storage,num_k);
        { // assemble 前向重算 ρ(q)(与 ⑥ 前向同表同式;⑥ 落码后可换其视图,位级一致)
            auto q_view=polar_rev_source_adjoint;
            auto rho_re=polar_rev_rho_fwd_re,rho_im=polar_rev_rho_fwd_im;
            auto dbre=polar_density_basis_re,dbim=polar_density_basis_im;
            auto kcos=polar_cosines,ksin=polar_sines;
            const Precision rho_scale=Precision(two_pi_cubed
                /polar_cell_volume);
            Kokkos::parallel_for("polar coulomb assemble",num_k,
                KOKKOS_LAMBDA(int k) {
                    Precision acc_cos[4],acc_sin[4];
                    for(int m=0;m<4;++m) {
                        acc_cos[m]=Precision(0);acc_sin[m]=Precision(0);
                    }
                    for(int node=0;node<owned_nodes;++node) {
                        const Precision c=kcos(k,node);
                        const Precision s=ksin(k,node);
                        for(int m=0;m<4;++m) {
                            const Precision q=q_view(node,m);
                            acc_cos[m]+=c*q;acc_sin[m]+=s*q;
                        }
                    }
                    Precision re=Precision(0),im=Precision(0);
                    for(int m=0;m<4;++m) {
                        re+=dbre(k,m)*acc_cos[m]+dbim(k,m)*acc_sin[m];
                        im+=dbim(k,m)*acc_cos[m]-dbre(k,m)*acc_sin[m];
                    }
                    rho_re(k)=rho_scale*re;rho_im(k)=rho_scale*im;
                });
            polar_global_sum_view(rho_re);
            polar_global_sum_view(rho_im);
        }
        { // ρ̄(含 V 因子与 kfac_c;k=0 处 kfac=0)
            auto rho_bar_re=polar_rev_rho_adj_re,rho_bar_im=polar_rev_rho_adj_im;
            auto rho_re=polar_rev_rho_fwd_re,rho_im=polar_rev_rho_fwd_im;
            auto kn2=polar_k_norm2,k0m=polar_k0_mask;
            auto kb_kweight=polar_k_weight;   // P4 RBE: 确定性=1
            const Precision rho_bar_scale=Precision(
                2.0*polar_cell_volume/(two_pi_cubed*two_pi_cubed)
                    *polar_field_constant);
            Kokkos::parallel_for("polar coulomb rho adjoint",num_k,
                KOKKOS_LAMBDA(int k) {
                    const Precision kfac=(k0m(k)>Precision(0))
                        ?Precision(0):Precision(1)/kn2(k);
                    const Precision w=kb_kweight(k);   // P4 RBE: 确定性=1
                    rho_bar_re(k)=rho_re(k)*kfac*rho_bar_scale*w;
                    rho_bar_im(k)=rho_im(k)*kfac*rho_bar_scale*w;
                });
        }
        kspace_assemble_reverse(polar_rev_rho_adj_re,
            polar_rev_rho_adj_im,polar_rev_source_adjoint,0,
            polar_rev_channel_adjoint,0);
        { // q̄ 拆分到两自旋通道(sum dim=1 的伴随)
            auto dbar=polar_rev_density_adjoint;
            auto q_bar=polar_rev_channel_adjoint;
            Kokkos::parallel_for("polar coulomb q adjoint split",num_nodes,
                KOKKOS_LAMBDA(int node) {
                    for(int m=0;m<4;++m) {
                        const Precision value=q_bar(node,m);
                        Kokkos::atomic_add(&dbar(node,m),value);
                        Kokkos::atomic_add(&dbar(node,4+m),value);
                    }
                });
        }
    }

        symmetrix_polar_profile_mark("rev5:scf-reverse");
// ---------- 5) SCF 逐步反向(step = M-1 … 0) ----------
    // [dbg v24] stage 检查点开关;生产运行不置位 ⇒ 零开销。
    const bool dbg_ckpt=(std::getenv("POLAR_DBG_CKPT")!=nullptr);
    if(dbg_ckpt) {
        ensure_view(polar_dbg_rev_dfinal,polar_dbg_rev_dfinal_storage,
            num_nodes,8);
        ordered_kokkos_deep_copy(polar_dbg_rev_dfinal,
            polar_rev_density_adjoint);
        ensure_view(polar_dbg_rev_pos_coul,polar_dbg_rev_pos_coul_storage,
            num_nodes,3);
        ordered_kokkos_deep_copy(polar_dbg_rev_pos_coul,
            polar_rev_position_adjoint);
    }
    ensure_view(polar_rev_fmix_adjoint,polar_rev_fmix_adjoint_storage,
        num_nodes,feature_width);
    ordered_kokkos_deep_copy(polar_rev_fmix_adjoint,Precision(0));
    // Plan B1 switch (default 0 = legacy replay).  The cache is populated by
    // the forward of the same evaluation; refuse a stale/missing cache.
    const bool replay_cache_on=polar_replay_cache_enabled();
    if(replay_cache_on&&!polar_rev_cache_valid)
        throw std::runtime_error(
            "MATOS_REPLAY_CACHE=1 but the forward SCF step cache is not "
            "populated (run_polar_pipeline must run first).");
    for(int step=polar_recursion_steps-1;step>=0;--step) {
        auto& update=polar_updates[step];
        const int hidden=update.nl_hidden_width;
        const int dp_width=update.source_embedding.output_dimension();
        const int gate_w=update.gate_scalar_count+3*update.gate_gated_count;
        const int ro_w=update.readout_linear_1.output_dimension();
        const int pe_w=update.potential_linear.output_dimension();
        // d_in(i): i=0 → 初始校正后密度; i>0 → i-1 步校后密度快照。
        // [v3 适配·D-04] i>0 直接消费既有 polar_scf_step_densities——其
        // "polar scf density snapshot" 内核位于逐步 Fukui 校正之后(校后口径, 与提案
        // step_densities_post 同值同源); 原提案的 _post 重复视图不新增。
        Kokkos::View<const Precision**,Kokkos::LayoutRight> density_in=
            step==0?Kokkos::View<const Precision**,Kokkos::LayoutRight>(
                        polar_field_independent_density)
                   :Kokkos::View<const Precision**,Kokkos::LayoutRight>(
                        Kokkos::subview(polar_scf_step_densities,
                            step-1,Kokkos::ALL,Kokkos::ALL));
        if(replay_cache_on) load_step_cache(step);
        else replay_step_forward(step,density_in);
        symmetrix_polar_profile_mark(
            replay_cache_on?"rev5a:cache-load":"rev5a:replay");
        ensure_view(polar_rev_channel_adjoint,polar_rev_channel_adjoint_storage,
            num_nodes,8);
        ordered_kokkos_deep_copy(polar_rev_channel_adjoint,Precision(0));
        // (a) 逐步 Fukui 校正反向:d̄(校正后) → d̄_mid + s̄。
        // [修正 v23 2026-10-16] head 必须取 fukui_reverse **之后** 的 dbar:
        //   前向 = [密度更新 d_mid=d_pre+out[:8]] → [校正 d_out=d_mid+src·f,
        //   pred=Σd_mid(镜像在密度更新后取)] ⇒ out̄[:8]=∂/∂out=d̄_mid=d̄_out−f̄
        //   (l=0 列), dbar 终点=∂/∂d_pre=d̄_mid。原实现先复制后调整 ⇒ out̄[:8]
        //   漏 −f̄ 全局项(f̄=Σd̄·s/S ≈ 种子量级, 帧相关)——d 通道偏大 2~9×的根因。
        ensure_view(polar_rev_out_adjoint,polar_rev_out_adjoint_storage,
            num_nodes,static_cast<int>(update.readout_linear_2.output_dimension()));
        ensure_view(polar_rev_fukui_source_adjoint,
            polar_rev_fukui_source_adjoint_storage,num_nodes,2);
        fukui_reverse(polar_rev_density_adjoint,polar_rev_out,/*source_offset=*/8,
            polar_scf_fukui_scalars.at(step),polar_rev_fukui_source_adjoint);
        {
            auto dbar=polar_rev_density_adjoint;
            auto out_adj=polar_rev_out_adjoint;
            Kokkos::parallel_for("polar rev out adjoint head",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{num_nodes,8}),
                KOKKOS_LAMBDA(int node,int column) {
                    out_adj(node,column)=dbar(node,column);
                });
        }
        {
            auto out_adj=polar_rev_out_adjoint;
            auto s_bar=polar_rev_fukui_source_adjoint;
            Kokkos::parallel_for("polar rev out adjoint tail",num_nodes,
                KOKKOS_LAMBDA(int node) {
                    out_adj(node,8)=s_bar(node,0);
                    out_adj(node,9)=s_bar(node,1);
                });
        }
        // [dbg v24] ∂E/∂out 检查点(末步=校正反向后全列伴)。
        if(dbg_ckpt) {
            if(step==polar_recursion_steps-1) {
                ensure_view(polar_dbg_rev_out_last,
                    polar_dbg_rev_out_last_storage,num_nodes,
                    static_cast<int>(polar_rev_out_adjoint.extent(1)));
                ordered_kokkos_deep_copy(polar_dbg_rev_out_last,
                    polar_rev_out_adjoint);
            }
            if(step==0) {
                ensure_view(polar_dbg_rev_out_first,
                    polar_dbg_rev_out_first_storage,num_nodes,
                    static_cast<int>(polar_rev_out_adjoint.extent(1)));
                ordered_kokkos_deep_copy(polar_dbg_rev_out_first,
                    polar_rev_out_adjoint);
            }
        }
        // (b) 更新块反向: out̄ → gate2 → ro2 → gate1 → ro1 → tp → MLP → nl_in → dp/se
        //     → pe → P̄ / d̄_ce / fmix̄。
        ensure_view(polar_rev_gate2_adjoint,polar_rev_gate2_adjoint_storage,
            num_nodes,gate_w);
        ensure_view(polar_rev_ro2_adjoint,polar_rev_ro2_adjoint_storage,
            num_nodes,ro_w);
        ensure_view(polar_rev_gate1_adjoint,polar_rev_gate1_adjoint_storage,
            num_nodes,gate_w);
        ensure_view(polar_rev_ro1_adjoint,polar_rev_ro1_adjoint_storage,
            num_nodes,ro_w);
        ensure_view(polar_rev_tp_adjoint,polar_rev_tp_adjoint_storage,
            num_nodes,static_cast<int>(polar_rev_tp.extent(1)));
        ensure_view(polar_rev_nlout_adjoint,polar_rev_nlout_adjoint_storage,
            num_nodes,static_cast<int>(polar_rev_nl_out.extent(1)));
        ensure_view(polar_rev_nl_adjoint,polar_rev_nl_adjoint_storage,
            num_nodes,3*hidden);
        ensure_view(polar_rev_h_adjoint,polar_rev_h_adjoint_storage,
            num_nodes,3*hidden);
        ensure_view(polar_rev_nlin_adjoint,polar_rev_nlin_adjoint_storage,
            num_nodes,static_cast<int>(polar_rev_nl_in.extent(1)));
        ensure_view(polar_rev_dp_adjoint,polar_rev_dp_adjoint_storage,
            num_nodes,dp_width);
        ensure_view(polar_rev_se_adjoint,polar_rev_se_adjoint_storage,
            num_nodes,dp_width);
        ensure_view(polar_rev_pe_adjoint,polar_rev_pe_adjoint_storage,
            num_nodes,pe_w);
        ensure_view(polar_rev_potential_adjoint,
            polar_rev_potential_adjoint_storage,num_nodes,16);
        update.readout_linear_2.reverse(polar_rev_out_adjoint,
            polar_rev_gate2_adjoint);
        ordered_kokkos_deep_copy(polar_rev_ro2_adjoint,Precision(0));
        readout_gate_reverse(polar_rev_ro2,polar_rev_gate2_adjoint,
            polar_rev_ro2_adjoint,update.gate_scalar_cst,
            update.gate_gate_cst,update.gate_scalar_count,
            update.gate_gate_count);
        update.readout_linear_mid.reverse(polar_rev_ro2_adjoint,
            polar_rev_gate1_adjoint);
        ordered_kokkos_deep_copy(polar_rev_ro1_adjoint,Precision(0));
        readout_gate_reverse(polar_rev_ro1,polar_rev_gate1_adjoint,
            polar_rev_ro1_adjoint,update.gate_scalar_cst,
            update.gate_gate_cst,update.gate_scalar_count,
            update.gate_gate_count);
        update.readout_linear_1.reverse(polar_rev_ro1_adjoint,
            polar_rev_tp_adjoint);
        symmetrix_polar_profile_mark("rev5b:readout-linears");
        { // (i) uvu_tp 反向: x1=fmix(→fmix̄), x2=nl_out(→nl̄_out)
            const auto to_host_vector=[&](const auto& view) {
                const auto mirror=Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(),view);
                std::vector<typename decltype(mirror)::value_type>
                    values(mirror.size());
                for(std::size_t i2=0;i2<values.size();++i2)
                    values[i2]=mirror.data()[i2];
                return values;
            };
            const std::vector<int> tp_meta=to_host_vector(update.uvu_tp_meta);
            const std::vector<Precision> tp_pw=
                to_host_vector(update.uvu_tp_path_weight);
            ordered_kokkos_deep_copy(polar_rev_nlout_adjoint,Precision(0));
            uvu_reverse(tp_meta,tp_pw,update.uvu_tp_weight,
                update.uvu_tp_mask,polar_features_mixed,polar_rev_nl_out,
                polar_rev_tp_adjoint,polar_rev_fmix_adjoint,
                polar_rev_nlout_adjoint);
        }
        symmetrix_polar_profile_mark("rev5c:uvu-tp");
        { // (ii) MLP 反向滚动: a 伴缓冲=nl_adjoint,h 伴缓冲=h_adjoint
            auto h_adj=polar_rev_h_adjoint;
            auto a_adj=polar_rev_nl_adjoint;
            auto h=polar_rev_nl_h;auto a=polar_rev_nl_a;
            // dense3 的输入伴写 a2 块([2h,3h)):以子视图承接(initialize 只清零该块)。
            dense_reverse(update.nl_w3,polar_rev_nlout_adjoint,
                Kokkos::subview(a_adj,Kokkos::ALL,
                    std::make_pair(2*hidden,3*hidden)),true);
            {
                auto h2=Kokkos::subview(h,Kokkos::ALL,
                    std::make_pair(2*hidden,3*hidden));
                auto a2=Kokkos::subview(a,Kokkos::ALL,
                    std::make_pair(2*hidden,3*hidden));
                auto a2_adj=Kokkos::subview(a_adj,Kokkos::ALL,
                    std::make_pair(2*hidden,3*hidden));
                auto h2_adj=Kokkos::subview(h_adj,Kokkos::ALL,
                    std::make_pair(2*hidden,3*hidden));
                ordered_kokkos_deep_copy(h2_adj,Precision(0));
                ln_silu_reverse(h2,a2,a2_adj,h2_adj,update.ln2_weight,
                    update.ln2_bias,update.ln_eps);
            }
            {
                auto h2_adj=Kokkos::subview(h_adj,Kokkos::ALL,
                    std::make_pair(2*hidden,3*hidden));
                auto a1_adj=Kokkos::subview(a_adj,Kokkos::ALL,
                    std::make_pair(hidden,2*hidden));
                dense_reverse(update.nl_w2,h2_adj,a1_adj,true);
            }
            {
                auto h1=Kokkos::subview(h,Kokkos::ALL,
                    std::make_pair(hidden,2*hidden));
                auto a1=Kokkos::subview(a,Kokkos::ALL,
                    std::make_pair(hidden,2*hidden));
                auto a1_adj=Kokkos::subview(a_adj,Kokkos::ALL,
                    std::make_pair(hidden,2*hidden));
                auto h1_adj=Kokkos::subview(h_adj,Kokkos::ALL,
                    std::make_pair(hidden,2*hidden));
                ordered_kokkos_deep_copy(h1_adj,Precision(0));
                ln_silu_reverse(h1,a1,a1_adj,h1_adj,update.ln1_weight,
                    update.ln1_bias,update.ln_eps);
            }
            {
                auto h1_adj=Kokkos::subview(h_adj,Kokkos::ALL,
                    std::make_pair(hidden,2*hidden));
                auto a0_adj=Kokkos::subview(a_adj,Kokkos::ALL,
                    std::make_pair(0,hidden));
                dense_reverse(update.nl_w1,h1_adj,a0_adj,true);
            }
            {
                auto h0=Kokkos::subview(h,Kokkos::ALL,
                    std::make_pair(0,hidden));
                auto a0=Kokkos::subview(a,Kokkos::ALL,
                    std::make_pair(0,hidden));
                auto a0_adj=Kokkos::subview(a_adj,Kokkos::ALL,
                    std::make_pair(0,hidden));
                auto h0_adj=Kokkos::subview(h_adj,Kokkos::ALL,
                    std::make_pair(0,hidden));
                ordered_kokkos_deep_copy(h0_adj,Precision(0));
                ln_silu_reverse(h0,a0,a0_adj,h0_adj,update.ln0_weight,
                    update.ln0_bias,update.ln_eps);
            }
            {
                auto h0_adj=Kokkos::subview(h_adj,Kokkos::ALL,
                    std::make_pair(0,hidden));
                dense_reverse(update.nl_w0,h0_adj,
                    polar_rev_nlin_adjoint,true);
            }
        }
        symmetrix_polar_profile_mark("rev5d:mlp-reverse");
        { // (iii) nl_in 伴拆分: [dp | se](se=attrs 常量路径,丢弃)
            auto nlin_adj=polar_rev_nlin_adjoint;
            auto dp_adj=polar_rev_dp_adjoint;
            auto se_adj=polar_rev_se_adjoint;
            Kokkos::parallel_for("polar rev nl_in split",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                    {num_nodes,dp_width}),
                KOKKOS_LAMBDA(int node,int column) {
                    dp_adj(node,column)=nlin_adj(node,column);
                    se_adj(node,column)=nlin_adj(node,dp_width+column);
                });
        }
        { // (iv) uvu_dot 反向: x1=fmix(→fmix̄), x2=pe(→pe伴)
            const auto to_host_vector=[&](const auto& view) {
                const auto mirror=Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(),view);
                std::vector<typename decltype(mirror)::value_type>
                    values(mirror.size());
                for(std::size_t i2=0;i2<values.size();++i2)
                    values[i2]=mirror.data()[i2];
                return values;
            };
            const std::vector<int> dot_meta=
                to_host_vector(update.uvu_dot_meta);
            const std::vector<Precision> dot_pw=
                to_host_vector(update.uvu_dot_path_weight);
            ordered_kokkos_deep_copy(polar_rev_pe_adjoint,Precision(0));
            uvu_reverse(dot_meta,dot_pw,update.uvu_dot_weight,
                update.uvu_dot_mask,polar_features_mixed,polar_rev_pe,
                polar_rev_dp_adjoint,polar_rev_fmix_adjoint,
                polar_rev_pe_adjoint);
        }
        { // (v) 势嵌入反: pe伴 → P̄ / fmix̄(累加) / d̄_ce(累加)
            update.potential_linear.reverse(polar_rev_pe_adjoint,
                polar_rev_potential_adjoint);
            update.node_feats_linear.reverse(polar_rev_pe_adjoint,
                polar_rev_fmix_adjoint,/*initialize_input_adjoint=*/false);
            update.charge_embedding.reverse(polar_rev_pe_adjoint,
                polar_rev_channel_adjoint,/*initialize_input_adjoint=*/false);
        }
        symmetrix_polar_profile_mark("rev5e:embedding-reverse");
        // (vi) 末步并入 ⑦ 的 ∂E/∂P_last;描述符反向(逐通道)。
        if(step==polar_recursion_steps-1&&(seed_terms&1)) {
            auto pbar=polar_rev_potential_adjoint;
            auto seed=polar_seed_potential_adjoint;
            Kokkos::parallel_for("polar rev last step P seed",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                    {num_nodes,16}),
                KOKKOS_LAMBDA(int node,int column) {
                    pbar(node,column)+=seed(node,column);
                });
        }
        ensure_view(polar_rev_feats_adjoint,polar_rev_feats_adjoint_storage,
            num_nodes,8);
        {
            auto pbar=polar_rev_potential_adjoint;
            auto feats_bar=polar_rev_feats_adjoint;
            auto knorm=polar_field_feature_norms;
            for(int channel=0;channel<2;++channel) {
                const int base=8*channel;
                Kokkos::parallel_for("polar rev feats split",num_nodes,
                    KOKKOS_LAMBDA(int node) {
                        for(int j=0;j<8;++j)
                            feats_bar(node,j)=pbar(node,base+j)/knorm(j);
                    });
                descriptor_reverse_channel(channel,density_in);
            }
        }
        symmetrix_polar_profile_mark("rev5f:descriptor-reverse");
        // (vii) d̄_i = d̄_pre + q̄(两通道) + d̄_ce
        {
            auto dbar=polar_rev_density_adjoint;
            auto extra=polar_rev_channel_adjoint;
            Kokkos::parallel_for("polar rev density chain merge",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{num_nodes,8}),
                KOKKOS_LAMBDA(int node,int column) {
                    dbar(node,column)+=extra(node,column);
                });
        }
        // [dbg v24] ∂E/∂d_post 检查点(末步迭代末 = 步1 输入密度全伴)。
        if(dbg_ckpt&&step==polar_recursion_steps-1) {
            ensure_view(polar_dbg_rev_dpost,polar_dbg_rev_dpost_storage,
                num_nodes,8);
            ordered_kokkos_deep_copy(polar_dbg_rev_dpost,
                polar_rev_density_adjoint);
        }
    }
    symmetrix_polar_profile_mark("rev5g:step-end");
    // ⑦ 的 ∂E/∂d_init(初校正后密度)在 SCF 链尾并入。
    if(seed_terms&1) {
        auto dbar=polar_rev_density_adjoint;
        auto seed_init=polar_seed_density0_adjoint;
        Kokkos::parallel_for("polar reverse seed density init",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{num_nodes,8}),
            KOKKOS_LAMBDA(int node,int column) {
                dbar(node,column)+=seed_init(node,column);
            });
    }
    // [dbg v24] ∂E/∂d_init(全种子)+ ∂E/∂features_mixed 检查点。
    if(dbg_ckpt) {
        ensure_view(polar_dbg_rev_dinit,polar_dbg_rev_dinit_storage,
            num_nodes,8);
        ordered_kokkos_deep_copy(polar_dbg_rev_dinit,
            polar_rev_density_adjoint);
        ensure_view(polar_dbg_rev_fmix,polar_dbg_rev_fmix_storage,
            num_nodes,feature_width);
        ordered_kokkos_deep_copy(polar_dbg_rev_fmix,
            polar_rev_fmix_adjoint);
    }

        symmetrix_polar_profile_mark("rev6:fukui-readout");
// ---------- 6) 初始化 Fukui 反向 → s̄0 → Readout::reverse_all → ⑧-A → 末层 ----------
    {
        ensure_view(polar_rev_fukui_source_adjoint,
            polar_rev_fukui_source_adjoint_storage,num_nodes,2);
        fukui_reverse(polar_rev_density_adjoint,polar_fukui_sources,
            /*source_offset=*/0,polar_fukui_init_scalars,
            polar_rev_fukui_source_adjoint);
        ensure_view(polar_rev_nfeat_adjoint,polar_rev_nfeat_adjoint_storage,
            num_nodes,feature_width);
        polar_fukui_source_map->reverse_all(polar_rev_fukui_source_adjoint,
            polar_rev_nfeat_adjoint,/*initialize_input_adjoint=*/true);
        ensure_view(states.back().layer_adjoint,
            states.back().layer_adjoint_storage,num_nodes,feature_width);
        basis_adjoint_add(polar_rev_nfeat_adjoint,
            states.back().layer_adjoint);
    }

        symmetrix_polar_profile_mark("rev7:lrsrc");
// ---------- 7) ⑧-1 lr_src 逐层反向(全量 d̄ 平权,sum 线性) ----------
    for(int layer=0;layer<layers;++layer) {
        ensure_view(states[layer].layer_adjoint,
            states[layer].layer_adjoint_storage,num_nodes,feature_width);
        polar_lr_source_maps[layer].reverse(polar_rev_density_adjoint,
            polar_rev_nfeat_adjoint,/*initialize_input_adjoint=*/true);
        basis_adjoint_add(polar_rev_nfeat_adjoint,
            states[layer].layer_adjoint);
    }

        symmetrix_polar_profile_mark("rev8:mixer");
// ---------- 8) ⑧-5 mixer 逐层反向 ----------
    for(int layer=0;layer<layers;++layer) {
        polar_mixer_linears[layer].reverse(polar_rev_fmix_adjoint,
            polar_rev_nfeat_adjoint,/*initialize_input_adjoint=*/true);
        basis_adjoint_add(polar_rev_nfeat_adjoint,
            states[layer].layer_adjoint);
    }

    // [dbg v24] 全链位置伴检查点(相位伴随总账;node_forces 写出前)。
    if(dbg_ckpt) {
        ensure_view(polar_dbg_rev_pos_all,polar_dbg_rev_pos_all_storage,
            num_nodes,3);
        ordered_kokkos_deep_copy(polar_dbg_rev_pos_all,
            polar_rev_position_adjoint);
    }

        symmetrix_polar_profile_mark("rev9:node-forces");
// ---------- 9) 位置伴 → 节点力(语义同 node_forces: −∂E/∂r) ----------
    {
        auto pos=polar_rev_position_adjoint;
        auto node_force=polar_node_forces;
        Kokkos::parallel_for("polar node force write",num_nodes,
            KOKKOS_LAMBDA(int node) {
                node_force(3*node)=-static_cast<double>(pos(node,0));
                node_force(3*node+1)=-static_cast<double>(pos(node,1));
                node_force(3*node+2)=-static_cast<double>(pos(node,2));
            });
    }

        symmetrix_polar_profile_mark("rev10:cleanup");
// ---------- 10) 收尾: 种子 consume-once ----------
    polar_reverse_seed_terms=0;
    if(polar_seed_density_adjoint.extent(0)!=0)
        ordered_kokkos_deep_copy(polar_seed_density_adjoint,Precision(0));
    if(polar_seed_density0_adjoint.extent(0)!=0)
        ordered_kokkos_deep_copy(polar_seed_density0_adjoint,Precision(0));
    if(polar_seed_potential_adjoint.extent(0)!=0)
        ordered_kokkos_deep_copy(polar_seed_potential_adjoint,Precision(0));
}

// ⑧-A 公开小工具(⑦ le 反向的 n̄f_last 路径投递口;hpp 声明在 §2.2)。
template<typename Precision>
void MaceNonlinearKokkosT<Precision>::polar_inject_node_features_adjoint(
    const int layer,Kokkos::View<const Precision**,Kokkos::LayoutRight> adjoint)
{
    if(!polar_enabled||layer<0||layer>=static_cast<int>(states.size()))
        throw std::invalid_argument(
            "polar node-feature adjoint injection layer is invalid.");
    if(static_cast<int>(adjoint.extent(1))!=polar_feature_width)
        throw std::invalid_argument(
            "polar node-feature adjoint width mismatch.");
    ensure_view(states[layer].layer_adjoint,states[layer].layer_adjoint_storage,
        adjoint.extent(0),polar_feature_width);
    auto source=polar_basis_source;
    auto sign=polar_basis_sign;
    auto internal=states[layer].layer_adjoint;
    const int num_nodes=static_cast<int>(adjoint.extent(0));
    Kokkos::parallel_for("polar inject node-feature adjoint",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
            {num_nodes,polar_feature_width}),
        KOKKOS_LAMBDA(int node,int column) {
            Kokkos::atomic_add(&internal(node,source(column)),
                sign(column)*adjoint(node,column));
        });
}

template class MaceNonlinearKokkosT<float>;
template class MaceNonlinearKokkosT<double>;
