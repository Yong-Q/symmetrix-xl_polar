# symmetrix-polar-les

Long-range electrostatics for the Symmetrix **polar** MACE path.

A polar MACE checkpoint predicts a per-atom latent charge from the local
neighbourhood.  This branch makes those charges interact through a periodic
Ewald sum — the **Latent Ewald Summation (LES)** route to MACELES — with
analytic forces and an analytic charge gradient, so the model is no longer
short-range only.

- base: upstream `6c5f73b`
- engine change: one commit, 9 files, +1,839 lines
- builds a LAMMPS pair style (`symmetrix/mace/float32/kk`) and the Symmetrix
  Python/ASE caller from the same tree
- upstream documentation: [UPSTREAM_README.md](UPSTREAM_README.md)

## What it adds

| file | role |
|---|---|
| `libsymmetrix/source/les_ewald.hpp` | periodic triclinic GTO-Ewald (σ = 1.0 Å, dl = 2.0 Å): energy, analytic forces, analytic `dE/dq`, host reference and a charge finite-difference self-test (double 5×10⁻¹², float 1.9×10⁻⁷) |
| `libsymmetrix/source/mace_kokkos.hpp` | `les_ewald_dE_dq`, readout-2 feature gradient, `apply_les_charge_adjoint()`, runtime `les_norm_factor` |
| `libsymmetrix/source/mace_kokkos_evaluate.cpp` | Ewald energy/forces and the charge adjoint immediately after every Ewald call, in both single- and two-layer paths |
| `libsymmetrix/source/mace_kokkos_second_interaction.cpp` | LES readouts (linear H1 + MLP H2), charge finalization, and the adjoint that chains `q_final = s·(q_raw − m·mean(q_raw))` into `H1_adj` / `H2_adj` |
| `libsymmetrix/source/mace_kokkos_model.cpp` | reads `les_norm_factor` from the model JSON instead of a hard-coded constant |
| `pair_symmetrix/pair_symmetrix_mace_kokkos.{cpp,h}` | LAMMPS box/position plumbing, Ewald force merge, optional per-atom charge dump |
| `symmetrix/source/symmetrix/extract_maceles.py` | checkpoint export: fusion basis, e3nn path normalization, `les_norm_factor`, `MACE_LES` model type |

The force is the full derivative

```
F_i = −∂E_local/∂r_i  − ∂E_LR/∂r_i|_q  − Σ_j (∂E_LR/∂q_j)(∂q_j/∂r_i)
```

The last term is the charge-response contribution a fixed-charge Ewald
implementation misses; without it the LES force increment is only 19–54 % of
the reference at 1,002–1,500 atoms.

## Measured speed

Clean A100 80 GB, FP32, `streamed_edges direct`, 15 × 20-step NVT samples per
arm (µs/atom/step, min / median):

| atoms | density (molecules/Å³) | MACELES (LES on) | polar backbone (LES off) | LES overhead |
|---:|---:|---:|---:|---:|
| 1,002 | 0.020 | 5.09 / 5.42 | 3.66 / 4.01 | +39 % |
| 1,500 | 0.0334 (water, 1 g/cm³) | **5.37 / 5.71** | 4.46 / 5.40 | +20 % |
| 5,001 | 0.020 | **4.22 / 4.23** | 2.73 / 2.79 | +54 % |

Six 3,000-atom boxes, single 100-step timings:

| system | H₂O | NH₃ | CH₄ | C₂H₄ | C₂H₆ | CO₂ |
|---|---:|---:|---:|---:|---:|---:|
| µs/atom/step | 7.93 | 7.90 | 7.00 | 7.05 | 7.24 | 6.60 |

Sustained: a 3,000-atom water box (1,000 molecules, 0.0334 Å⁻³, 300 K, 0.5 fs)
runs at 22.88 ms/step = **7.63 µs/atom/step**, i.e. **1 ns ≈ 12.7 h** on one
A100.  The 1,500-atom ambient box reaches 1 ns in ≈4.5 h clean (5.3 h
sustained).  The LES share grows with box volume (reciprocal-space Ewald work)
and shrinks with neighbour density, hence +20 % at 1,500 ambient atoms versus
+54 % at 5,001 low-density atoms.

## Accuracy against the PyTorch reference

Same checkpoint, FP32 engine vs FP32/FP64 PyTorch, identical geometries:

| system | atoms | F MAE (eV/Å) | F max (eV/Å) | corr |
|---|---:|---:|---:|---:|
| box | 1,002 | 3.0×10⁻⁵ | 1.6×10⁻⁴ | 0.999999 |
| lattice | 1,500 | 2.4×10⁻⁵ | 1.3×10⁻⁴ | 1.000000 |
| H₂O gas | 96 | 4.4×10⁻⁵ | 2.1×10⁻⁴ | 1.000000 |
| NH₃ gas | 128 | 3.7×10⁻⁵ | 2.5×10⁻⁴ | 1.000000 |
| CH₄ gas | 160 | 2.3×10⁻⁵ | 2.1×10⁻⁴ | 1.000000 |
| C₂H₄ gas | 192 | 3.0×10⁻⁵ | 2.5×10⁻⁴ | 1.000000 |
| C₂H₆ gas | 256 | 2.2×10⁻⁵ | 1.6×10⁻⁴ | 1.000000 |
| CO₂ gas | 96 | 1.4×10⁻⁴ | 6.9×10⁻⁴ | 1.000000 |

Energies: −3×10⁻⁶ meV/atom (1,002 atoms), −7×10⁻⁶ meV/atom (1,500 atoms);
gas molecules agree to ≤1.5×10⁻³ meV/atom.

## Build

```bash
git clone --recurse-submodules -b polar-les \
    git@github.com:Yong-Q/symmetrix-xl_polar.git
cd symmetrix-xl

# LAMMPS pair style
./pair_symmetrix/install.sh <lammps-source>
cmake -S <lammps-source>/cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_MPI=OFF -DPKG_KOKKOS=ON -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ENABLE_SERIAL=ON -DCMAKE_CXX_STANDARD=20 \
      -DCMAKE_CXX_COMPILER=<lammps>/lib/kokkos/bin/nvcc_wrapper \
      -DCMAKE_CUDA_COMPILER=<cuda>/bin/nvcc \
      -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j
```

`streamed_edges direct` requires a device artifact for the checkpoint, built on
the target GPU:

```bash
python -m symmetrix.cli.prepare_jit_device_artifact \
    --model maceles.json --precision float32 --lammps-arguments
```

LAMMPS side:

```lammps
pair_style symmetrix/mace/float32/kk no_domain_decomposition \
    streamed_edges direct profile speed \
    jit_device_artifact <module.cubin> jit_device_blocks_per_compute_unit 8
```

## Keeping up with upstream

`polar-les` is one commit on `6c5f73b`.  To follow upstream:

```bash
git remote add upstream https://github.com/bonan-group/symmetrix-xl.git
git fetch upstream && git rebase upstream/main
```

A rebase only touches the 9 LES files; re-run the force/energy comparison
above whenever `les_ewald.hpp`, `mace_kokkos*`, `pair_symmetrix_*` or
`extract_maceles.py` change.
