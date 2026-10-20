"""Serialization of PolarMACE electrostatic ("polar") blocks into the MACE_Nonlinear JSON.

依据 (all grounded in upstream sources):
- mace.modules.field_blocks: MultiLayerFeatureMixer / EnvironmentDependentSpinSourceBlock /
  PotentialEmbeddingBlock / AgnosticChargeBiasedLinearPotentialEmbedding / FieldUpdateBlock /
  AgnosticEmbeddedOneBodyVariableUpdate / SparseUvuTensorProduct / PostScfReadout /
  OneBodyMLPFieldReadout  —— 逐 class 按其 state_dict / attribute 结构序列化;
- graph_longrange (v0.4.0): GTOElectrostaticFeatures / GTOElectrostaticEnergy / GTOBasis /
  GTOSelfInteractionBlock / DisplacedGTOExternalFieldBlock —— 按模块属性/state_dict 序列化;
- 原语词汇表与既有 backbone 序列化一致 (e3nn Linear / uvu TensorProduct / gate / MLP),
  引擎侧已有同族实现 (E3Linear/E3ProductBasis/Gate/AffineMLP),故可直接消费。

未识别到结构化序列化器的叶子模块:给出类名 + 原始 state_dict 张量 (raw),并记入
``unserialized`` 清单 —— 诚实标记,待引擎按模块逐个实现后升级为结构化 schema。
"""

from __future__ import annotations

from typing import Any

import torch

from .extract_mace_nonlinear import (
    _instructions,
    _linear,
    _matrix_tensor,
    _readout,
    _tensor,
)


def _linear_or_none(mod) -> Any:
    return _linear(mod) if mod is not None else None


def _sparse_uvu(tp) -> dict[str, Any]:
    """SparseUvuTensorProduct: e3nn TensorProduct 参数布局 (weight + output_mask)。

    与 backbone ``_tensor_product`` 同构 (instructions/weight/output_mask);
    连接模式固定 uvu (类内校验), 引擎既有 e3nn 张量积机制可直接消费。
    """
    out: dict[str, Any] = {
        "kind": "sparse_uvu",
        "irreps_in1": str(tp.irreps_in1),
        "irreps_in2": str(tp.irreps_in2),
        "irreps_out": str(tp.irreps_out),
        "layout": getattr(tp, "layout", "mul_ir"),
        "weight": _tensor(tp.weight),
        "output_mask": _tensor(tp.output_mask),
        "instructions": [],
    }
    for ins in tp.instructions:
        out["instructions"].append({
            "i_in1": int(ins.i_in1),
            "i_in2": int(ins.i_in2),
            "i_out": int(ins.i_out),
            "connection_mode": ins.connection_mode,
            "has_weight": bool(ins.has_weight),
            "path_shape": list(ins.path_shape),
            # G1(审计): 前向逐路径乘 path_weight(field_blocks.py L452/L460),
            # 旧版漏存 → 引擎按 1.0 误算。来源 ins.path_weight(e3nn 指令)。
            "path_weight": float(getattr(ins, "path_weight", 1.0)),
        })
    return out


def _torch_mlp_net(net) -> dict[str, Any]:
    """Generic torch Sequential (Linear / e3nn-Activation-like / SiLU / identity)."""
    layers = []
    for layer in net:
        cname = type(layer).__name__
        if cname == "Linear":
            layers.append({
                "type": "linear",
                "weight": _tensor(layer.weight),
                "bias": _tensor(layer.bias) if layer.bias is not None else None,
            })
        elif cname in ("SiLU", "ReLU", "Tanh"):
            layers.append({"type": cname.lower()})
        elif cname == "Activation":
            w = getattr(layer, "weight", None)
            layers.append({
                "type": "activation",
                "weight": _tensor(w) if w is not None else None,
                "acts": [type(a).__name__.lower() for a in getattr(layer, "acts", [])],
            })
        elif cname in ("Identity", "NoNonLinearity"):
            layers.append({"type": "identity"})
        elif cname == "LayerNorm":
            # 结构化(引擎需 eps + 仿射权重; 例如 charges_maps 的 RadialMLP 三层 LN)
            layers.append({
                "type": "layer_norm",
                "eps": float(layer.eps),
                "normalized_shape": list(layer.normalized_shape),
                "weight": _tensor(layer.weight) if layer.elementwise_affine else None,
                "bias": _tensor(layer.bias) if layer.elementwise_affine else None,
            })
        else:
            # 未知层类: 记录类名 + 原始参数 (诚实标记, 待引擎/序列化器升级)
            layers.append({
                "type": cname,
                "raw_state_dict": {k: _tensor(v) for k, v in layer.state_dict().items()},
            })
    return {"kind": "torch_mlp", "layers": layers}


def _gto_basis(b) -> dict[str, Any]:
    out: dict[str, Any] = {
        "kind": "gto_basis",
        "max_l": int(getattr(b, "max_l")),
        "sigmas": [float(s) for s in getattr(b, "sigmas")],
        "normalize": str(getattr(b, "normalize")),
        "use_spline": bool(getattr(b, "use_spline", True)),
        "kspace_cutoff": float(getattr(b, "kspace_cutoff")),
    }
    sd = b.state_dict()
    for key in ("cl_scale", "expanded_l_indices", "real_phase_factors",
                "imag_phase_factors", "permute_indices"):
        if key in sd:
            out[key] = _tensor(sd[key])
    if hasattr(b, "radial_spline") and b.radial_spline is not None:
        rs = b.radial_spline.state_dict()
        out["radial_spline"] = {k: _tensor(v) for k, v in rs.items()}
    return out


def _self_interaction(si) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": type(si).__name__}
    for k, v in si.state_dict().items():
        out[k.replace(".", "_")] = _tensor(v)
    return out


def _gto_features(mod) -> dict[str, Any]:
    out: dict[str, Any] = {
        "kind": type(mod).__name__,
        "density_basis": _gto_basis(mod.density_basis),
        "feature_basis": _gto_basis(mod.feature_basis),
        "include_self_interaction": bool(getattr(mod, "include_self_interaction", False)),
        "kspace_cutoff": float(getattr(mod, "kspace_cutoff")),
    }
    sd = mod.state_dict()
    if "output_permutation" in sd:
        out["output_permutation"] = _tensor(sd["output_permutation"])
    if hasattr(mod, "self_interaction_terms"):
        out["self_interaction_terms"] = _self_interaction(mod.self_interaction_terms)
    # 其余子模块 (realspace_features / non_periodic_correction_terms 等): 结构化+raw
    handled = {"density_basis", "feature_basis", "self_interaction_terms"}
    extra = {}
    for name, child in mod.named_children():
        if name in handled:
            continue
        extra[name] = {
            "kind": type(child).__name__,
            "raw_state_dict": {k: _tensor(v) for k, v in child.state_dict().items()},
            "attrs": {a: v for a in ("density_max_l", "projection_max_l", "density_smearing_width",
                                     "num_radial", "offset")
                      if isinstance(v := getattr(child, a, None), (int, float))},
        }
    out["aux_blocks"] = extra
    return out


def _gto_energy(mod) -> dict[str, Any]:
    out: dict[str, Any] = {
        "kind": type(mod).__name__,
        "density_basis": _gto_basis(mod.density_basis),
        "include_self_interaction": bool(getattr(mod, "include_self_interaction", False)),
        "include_pbc_corrections": bool(getattr(mod, "include_pbc_corrections", True)),
        "kspace_cutoff": float(getattr(mod, "kspace_cutoff")),
    }
    if hasattr(mod, "self_interaction_terms"):
        out["self_interaction_terms"] = _self_interaction(mod.self_interaction_terms)
    extra = {}
    for name, child in mod.named_children():
        if name in ("density_basis", "self_interaction_terms"):
            continue
        extra[name] = {
            "kind": type(child).__name__,
            "raw_state_dict": {k: _tensor(v) for k, v in child.state_dict().items()},
        }
    out["aux_blocks"] = extra
    return out


def _potential_embedding(pe) -> dict[str, Any]:
    return {
        "kind": type(pe).__name__,
        "potential_linear": _linear_or_none(getattr(pe, "potential_linear", None)),
        "node_feats_linear": _linear_or_none(getattr(pe, "node_feats_linear", None)),
        "charge_embedding": _linear_or_none(getattr(pe, "charge_embedding", None)),
    }


def _field_update_block(b) -> dict[str, Any]:
    out: dict[str, Any] = {
        "kind": type(b).__name__,
        "field_norm_factor": float(getattr(b, "field_norm_factor", 0.0)),
        "potential_embedding": _potential_embedding(b.potential_embedding),
        "source_embedding": _linear_or_none(getattr(b, "source_embedding", None)),
        "dot_products": _sparse_uvu(b.dot_products) if hasattr(b, "dot_products") else None,
        "tp_out": _sparse_uvu(b.tp_out) if hasattr(b, "tp_out") else None,
        "readout": _readout(b.readout) if hasattr(b, "readout") else None,
        "nonlinearity": _torch_mlp_net(b.nonlinearity.net)
        if hasattr(b, "nonlinearity") and hasattr(b.nonlinearity, "net") else None,
    }
    if hasattr(b, "readout") and hasattr(b.readout, "equivariant_nonlin"):
        # G2(审计): e3nn.nn.Gate 前向消费 act 常数(标量分支 ×cst_silu、门分支 ×cst_sigmoid)
        # 与 mul(ElementwiseTensorProduct, uuu) 指令;旧版仅 raw_state_dict(无 cst/指令)。
        # act 语义在 readout ctor 固定(blocks.py L331-332: scalars=silu, gates=sigmoid)。
        g = b.readout.equivariant_nonlin
        gate: dict[str, Any] = {
            "kind": type(g).__name__,
            "raw_state_dict": {
                k: _tensor(v) for k, v in g.state_dict().items()
            },
        }
        for key, hint in (("act_scalars", "silu"), ("act_gates", "sigmoid")):
            acts = getattr(g, key, None)
            act_list = getattr(acts, "acts", None) if acts is not None else None
            if act_list:
                gate[key] = [
                    {"name": hint, "cst": float(getattr(a, "cst", 1.0))}
                    for a in act_list
                ]
        for attr in ("irreps_scalars", "irreps_gates", "irreps_gated",
                     "irreps_in", "irreps_out"):
            v = getattr(g, attr, None)
            if v is not None:
                gate[attr] = str(v)
        mul = getattr(g, "mul", None)
        if mul is not None:
            mul_out: dict[str, Any] = {"kind": type(mul).__name__}
            for attr in ("irreps_in1", "irreps_in2", "irreps_out"):
                v = getattr(mul, attr, None)
                if v is not None:
                    mul_out[attr] = str(v)
            for attr in ("weight", "output_mask"):
                v = getattr(mul, attr, None)
                if v is not None:
                    mul_out[attr] = _tensor(v)
            try:
                mul_out["instructions"] = _instructions(mul)
            except Exception:  # noqa: BLE001 - 诚实降级: 结构不可得时保留其余字段
                pass
            gate["mul"] = mul_out
        out["readout_gate"] = gate
    return out


def _onebody_readout(m) -> dict[str, Any]:
    out: dict[str, Any] = {
        "kind": type(m).__name__,
        "linear_up_q": _linear_or_none(getattr(m, "linear_up_q", None)),
        "linear_up_v": _linear_or_none(getattr(m, "linear_up_v", None)),
        "dot_products_q": _sparse_uvu(m.dot_products_q) if hasattr(m, "dot_products_q") else None,
        "dot_products_v": _sparse_uvu(m.dot_products_v) if hasattr(m, "dot_products_v") else None,
    }
    if hasattr(m, "mlp") and hasattr(m.mlp, "net"):
        out["mlp"] = _torch_mlp_net(m.mlp.net)
    return out


def extract_polar_blocks(model) -> dict[str, Any] | None:
    """Dump the electrostatic block set (None when the model has no polar attrs)."""
    if not any(hasattr(model, a) for a in
               ("electric_potential_descriptor", "coulomb_energy",
                "field_dependent_charges_maps", "local_electron_energy")):
        return None
    out: dict[str, Any] = {}
    if hasattr(model, "lr_source_maps"):
        out["lr_source_maps"] = [_linear(m.linear) for m in model.lr_source_maps]
    if hasattr(model, "fukui_source_map"):
        out["fukui_source_map"] = _readout(model.fukui_source_map)
    if hasattr(model, "electric_potential_descriptor"):
        out["electric_potential_descriptor"] = _gto_features(model.electric_potential_descriptor)
    if hasattr(model, "coulomb_energy"):
        out["coulomb_energy"] = _gto_energy(model.coulomb_energy)
    if hasattr(model, "field_dependent_charges_maps"):
        out["field_dependent_charges_maps"] = [
            _field_update_block(b) for b in model.field_dependent_charges_maps
        ]
    if hasattr(model, "layer_feature_mixer"):
        out["layer_feature_mixer"] = {
            "kind": "MultiLayerFeatureMixer",
            "linears": [_linear(m) for m in model.layer_feature_mixer.linears],
        }
    if hasattr(model, "local_electron_energy"):
        out["local_electron_energy"] = _onebody_readout(model.local_electron_energy)
    if hasattr(model, "external_field_contribution"):
        member = next(iter(model.external_field_contribution.state_dict().values()), None)
        out["external_field_contribution"] = {
            "kind": "DisplacedGTOExternalFieldBlock",
            "raw_state_dict": {
                k: _tensor(v) for k, v in model.external_field_contribution.state_dict().items()
            },
        }
    # G4(审计): 布局/切法语义登记(JSON 张量无法表达的运行时约定;锚点均带行号)。
    out["semantics"] = {
        "local_charges_layout": "[s\u03b1,x\u03b1,y\u03b1,z\u03b1,s\u03b2,x\u03b2,y\u03b2,z\u03b2] (extensions.py L734 view(N,2,4))",
        "potential_features_layout": "cat(\u03b1[:, :8], \u03b2[:, :8]); \u5378\u5df2 /field_feature_norms (extensions.py L834-848)",
        "charges_out_split": "\u524d8=\u5bc6\u5ea6\u589e\u91cf(view N,2,4), \u540e2=Fukui \u6e90 (extensions.py L833-834)",
        "density_update": "density += sources; \u6bcf\u6b65\u91cd\u505a\u9010\u56fe\u5f52\u4e00+l=0 \u6821\u6b63 (extensions.py L836-866)",
        "energy_assembly_order": "e0+inter_e \u2192 +le_total \u2192 +electro \u2192 (+external\u00b7dipole); \u5747\u5728 scale/shift \u540e (extensions.py L866-929/936)",
        "le_inputs": "charges_0=\u521d Fukui \u6821\u6b63\u5bc6\u5ea6; charges_induced=SCF \u672b\u6001; field_feats=\u672b\u6b65\u52bf\u7279\u5f81 (\u8bc4\u4f30\u81ea\u8be5\u6b65\u8f93\u5165\u5bc6\u5ea6)",
    }
    out["configs"] = {
        "add_local_electron_energy": bool(getattr(model, "add_local_electron_energy", False)),
        "return_electrostatic_potentials": bool(
            getattr(model, "return_electrostatic_potentials", False)),
        # SCF 所需标量/常量(非 1 常数, 必须显式序列化; 依据 extensions.py 同名属性):
        "num_recursion_steps": int(getattr(model, "num_recursion_steps", 0)),
        "field_feature_norms": _tensor(model.field_feature_norms)
        if hasattr(model, "field_feature_norms") else None,
        "from_ell_max_field_update": int(
            getattr(model, "from_ell_max_field_update", 0)),
    }
    return out
