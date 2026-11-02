<p align="center">
  <b>symmetrix-polar-les</b><br>
  <sub>MACELES &mdash; MACE with Latent Ewald Summation &mdash; on the Symmetrix polar backbone</sub>
</p>

<p align="center">
  <a href="#build"><img alt="base" src="https://img.shields.io/badge/base-6c5f73b-blue"></a>
  <img alt="diff" src="https://img.shields.io/badge/patch-9%20files%20%2B1%2C839%20lines-lightgrey">
  <img alt="speed" src="https://img.shields.io/badge/A100-5--8%20%C2%B5s%2Fatom%2Fstep-brightgreen">
  <img alt="force" src="https://img.shields.io/badge/force%20MAE-2e--5%E2%80%931.4e--4%20eV%2F%C3%85-success">
  <img alt="corr" src="https://img.shields.io/badge/corr-%E2%89%A5%200.99999-informational">
  <img alt="license" src="https://img.shields.io/github/license/Yong-Q/symmetrix-xl_polar">
</p>

> [!IMPORTANT]
> A polar MACE checkpoint predicts a per-atom latent charge from the local
> neighbourhood, but keeps the interaction **short-ranged**. This branch adds the
> missing long range: those charges interact through a **periodic GTO-Ewald sum**,
> with **analytic forces that include the charge-response term**, and it runs at
> production speed inside LAMMPS.

## Highlights

| | |
|---|---|
| **Complete force** | local + fixed-charge Ewald + the charge-response adjoint `Σⱼ (∂E_LR/∂qⱼ)(∂qⱼ/∂rᵢ)` that a naive Ewald hookup drops |
| **Fast** | **5–8 µs/atom/step** on one A100 with LES on; overhead **+20 % … +54 %** over the polar backbone |
| **Accurate** | force MAE **2×10⁻⁵ … 1.4×10⁻⁴ eV/Å** vs PyTorch on identical geometries, correlation **≥ 0.99999** |
| **Small patch** | **one commit** on upstream `6c5f73b` — **9 files, +1,839 lines**, nothing outside the LES path |

<p align="center"><sub>Maintained by <a href="https://github.com/Yong-Q">@Yong-Q</a></sub></p>

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

The force is the **full derivative**:

$$
\mathbf{F}_i \;=\; -\frac{\partial E_{\text{local}}}{\partial \mathbf{r}_i}
\;-\;\left.\frac{\partial E_{\text{LR}}}{\partial \mathbf{r}_i}\right|_q
\;-\;\sum_j \frac{\partial E_{\text{LR}}}{\partial q_j}\,\frac{\partial q_j}{\partial \mathbf{r}_i}
$$

> [!WARNING]
> The last term is the charge-response contribution that a fixed-charge Ewald
> implementation misses. Without it the LES force increment is only **19–54 %**
> of the reference at 1,000–1,500 atoms — energies still match, forces do not.

## Measured speed

Clean A100 80 GB, FP32, `streamed_edges direct`, 15 x 20-step NVT samples per
arm (us/atom/step, min / median):

| atoms | MACELES (LES on) | backbone (LES off) | LES overhead |
|---:|---:|---:|---:|
| ~1,000 | 5.09 / 5.42 | 3.66 / 4.01 | +39 % |
| **~1,500** | **5.37 / 5.71** | 4.46 / 5.40 | **+20 %** |
| ~5,000 | **4.22 / 4.23** | 2.73 / 2.79 | +54 % |

3,000-atom boxes of several molecular species: **6.6-7.9 us/atom/step**
with LES on, i.e. the same ballpark as the smaller boxes.

Sustained run, 3,000-atom box, 0.5 fs: **22.9 ms/step = 7.6 us/atom/step**,
so **1 ns takes about 12.7 h on one A100**.  The 1,500-atom box reaches 1 ns in
about 4.5 h clean (5.3 h sustained).

> [!NOTE]
> The LES share grows with box volume (reciprocal-space Ewald work) and shrinks
> with neighbour density — hence **+20 %** on the small dense box versus
> **+54 %** on the low-density 5,000-atom box.

## Accuracy against the PyTorch reference

Same checkpoint, FP32 engine vs FP32/FP64 PyTorch, identical geometries, on
systems between 96 and 1,500 atoms (bulk boxes and isolated molecules):

| quantity | range |
|---|---|
| force MAE | **2.2 x 10^-5 - 1.4 x 10^-4 eV/A** |
| force max | 1.3 x 10^-4 - 6.9 x 10^-4 eV/A |
| force correlation | **>= 0.99999** |
| energy | <= 7 x 10^-6 meV/atom |

The largest deviation is on a small isolated molecule, where the per-atom
reference forces are ~1 eV/A.

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
