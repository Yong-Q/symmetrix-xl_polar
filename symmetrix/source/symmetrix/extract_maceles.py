#!/usr/bin/env python3
"""Export a MACELES checkpoint to a Symmetrix JSON with the LES readouts.

The stock ``extract_mace_data`` labels every non-MACEField checkpoint as
plain ``MACE`` and silently drops the latent-Ewald parts.  This module keeps
the MACE backbone export and adds a ``latent_ewald`` section describing

    q_i = output_scale * sum_l ( readout_l(features_l) )_i

plus the fixed Gaussian-Ewald parameters of ``les.module.ewald.Ewald``.

The engine still needs a ``MACE_LES`` forward branch; this file only makes
the exported model faithful and self-describing.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .extract_mace_data import extract_mace_data


def _dense(linear):
    """Effective e3nn Linear 0e->0e weight (mul_out, mul_in).

    e3nn Linear multiplies by 1/sqrt(fan_in); store the already-normalised
    matrix so the engine can use a plain matrix product.
    """
    w = linear.weight.detach().cpu().numpy()
    instructions = getattr(linear, "instructions", None)
    if instructions is not None and len(instructions) == 1:
        # e3nn stores the weight as (in_dim, out_dim) per path; the forward
        # normalises each path by 1/sqrt(in_dim), not by the total irreps.
        in_dim, out_dim = instructions[0].path_shape
        w = w.reshape(in_dim, out_dim).T
        mul_in = in_dim
    else:
        mul_out = linear.irreps_out.num_irreps
        mul_in = linear.irreps_in.num_irreps
        if w.ndim == 1:
            w = w.reshape(mul_out, mul_in)
    w = w / np.sqrt(mul_in)
    bias = linear.bias.detach().cpu().numpy() if linear.bias.numel() else None
    return w.astype(np.float32), (None if bias is None else bias.astype(np.float32))


def _activation_spec(module):
    """Describe the MACE normalise2mom activation: cst * silu(x)."""
    acts = getattr(module, "acts", None)
    if acts is None or len(acts) == 0:
        return {"activation": type(module).__name__.lower(), "scale": 1.0}
    a = acts[0]
    cst = float(getattr(a, "cst", 1.0))
    fn = getattr(a, "f", None)
    name = "silu"
    if fn is not None:
        try:
            probe = float(fn(torch.tensor(0.5, dtype=torch.float64)))
            if abs(probe - 0.5 / (1 + np.exp(-0.5))) < 1e-9:
                name = "silu"
        except Exception:
            pass
    return {"activation": name, "scale": cst,
            "normalization": type(a).__name__}


def _fusion_basis(model):
    """Channel-mixing matrix that the compact exporter folds into readout 1.

    The compact Symmetrix H1 basis is the fused basis of the second
    interaction's ``linear_up``.  The stock extractor converts the linear
    energy readout with ``inv(weights_to_fuse[0])``; the LES linear readout
    must be converted the same way.  The nonlinear readout consumes H2 and is
    left in the raw basis.
    """
    irreps = model.les_readouts[0].linear.irreps_in
    # The linear readout only uses the 0e block; take its multiplicity.
    channels = None
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            channels = int(mul)
            break
    if channels is None:
        channels = int(irreps.num_irreps)
    up = model.interactions[1].linear_up.weight.detach().cpu().numpy()
    up = up / np.sqrt(channels)
    n_blocks = up.size // (channels * channels)
    wf = up.reshape(n_blocks, channels, channels)
    return np.linalg.inv(wf[0])


def latent_ewald_section(model):
    les = model.les
    fusion = _fusion_basis(model)
    readouts = []
    for i, readout in enumerate(model.les_readouts):
        entry = {"input_layer": i}
        if hasattr(readout, "linear"):
            w, b = _dense(readout.linear)
            # H1 in the compact engine is fused: apply the same basis change
            # the stock exporter applies to readout_1_weights.
            w = fusion @ w.reshape(-1)
            entry.update({"kind": "linear", "weight": w.tolist(),
                          "bias": None if b is None else b.tolist()})
        elif hasattr(readout, "linear_1") and hasattr(readout, "linear_2"):
            w1, b1 = _dense(readout.linear_1)
            w2, b2 = _dense(readout.linear_2)
            entry.update({
                "kind": "mlp",
                **_activation_spec(readout.non_linearity),
                "w1": w1.tolist(), "b1": None if b1 is None else b1.tolist(),
                "w2": w2.tolist(), "b2": None if b2 is None else b2.tolist(),
            })
        else:
            raise RuntimeError(f"unsupported LES readout {type(readout).__name__}")
        readouts.append(entry)
    return {
        "combine": "sum_over_readouts_times_output_scale",
        "output_scale": float(getattr(model, "les_output_scale", 1.0)),
        "sigma_A": float(les.sigma),
        "dl_A": float(les.dl),
        "norm_factor": float(les.ewald.norm_factor),
        "remove_self_interaction": bool(les.ewald.remove_self_interaction),
        # Les.remove_mean only affects the BEC head; the Ewald energy in the
        # reference does not subtract the mean charge.
        "remove_mean": False,
        "bec_remove_mean": bool(les.remove_mean),
        "use_epsilon_r_scaling": bool(getattr(les, "use_epsilon_r_scaling", False)),
        "epsilon_factor": float(getattr(les, "epsilon_factor", 1.0)),
        "use_atomwise": bool(getattr(les, "use_atomwise", False)),
        "readouts": readouts,
    }


def extract_maceles_data(model_path, species=None, head=None,
                         num_spline_points=256, radial_format="compact"):
    base = extract_mace_data(model_path, species=species, head=head,
                             num_spline_points=num_spline_points,
                             radial_format=radial_format)
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    if type(model).__name__ != "MACELES":
        raise RuntimeError(f"not a MACELES checkpoint: {type(model).__name__}")
    base["model_type"] = "MACE_LES"
    les = latent_ewald_section(model)
    base["latent_ewald"] = les
    base["les_num_readouts"] = len(model.les_readouts)
    base["les_keep_last_layer_irreps"] = True

    # Flat keys for the native MACEKokkos parser (same layout as the MACE
    # energy readouts: linear weights + 2-layer MLP + scale factor).
    r0, r1 = les["readouts"]
    base["les_output_scale"] = les["output_scale"]
    base["les_sigma_A"] = les["sigma_A"]
    base["les_dl_A"] = les["dl_A"]
    base["les_norm_factor"] = les["norm_factor"]
    base["les_remove_self_interaction"] = les["remove_self_interaction"]
    base["les_remove_mean"] = les["remove_mean"]
    base["les_readout_1_weights"] = list(np.asarray(r0["weight"]).reshape(-1))
    base["les_readout_2_weights_1"] = list(np.asarray(r1["w1"]).reshape(-1))
    base["les_readout_2_weights_2"] = list(np.asarray(r1["w2"]).reshape(-1))
    base["les_readout_2_scale_factor"] = float(r1["scale"])
    base["les_readout_2_hidden_size"] = int(np.asarray(r1["w1"]).shape[0])
    base["les_activation"] = r1["activation"]
    base["les_activation_normalization"] = r1.get("normalization", "")
    return base


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "-m", required=True)
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--chemical-symbols", "-s", nargs="+", default=None)
    ap.add_argument("--head", "-H", default=None)
    ap.add_argument("--num-spline-points", type=int, default=256)
    ap.add_argument("--radial-format", default="compact",
                    choices=["compact", "original"])
    args = ap.parse_args(argv)

    data = extract_maceles_data(
        args.model, species=args.chemical_symbols, head=args.head,
        num_spline_points=args.num_spline_points,
        radial_format=args.radial_format)
    with open(args.output, "w") as fh:
        if args.radial_format == "compact":
            json.dump(data, fh, separators=(",", ":"))
        else:
            json.dump(data, fh, indent=4)
        fh.write("\n")
    print(f"wrote {args.output}: model_type={data['model_type']} "
          f"readouts={len(data['latent_ewald']['readouts'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
