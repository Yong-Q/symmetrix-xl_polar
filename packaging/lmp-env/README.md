# Portable MACELES / LAMMPS engine bundle

Unpack, source `env.sh`, run. No compiler, no CUDA install, no Python build step.

```bash
tar xzf maceles-lmp-env-<ver>.tar.gz && cd maceles-lmp-env-<ver>
. ./env.sh
examples/smoke/run.sh            # 64 waters, 500 steps, ~2 min on one A100
```

## What is inside

| Path | Contents |
|---|---|
| `bin/lmp` | LAMMPS `stable_30Sep2026` + Kokkos/CUDA + `pair_style symmetrix/mace/.../kk` (MACELES-LES) |
| `bin/lmp-run` | wrapper: sources `env.sh`, checks for missing shared libraries, then execs `lmp` |
| `lib/` | `libcudart.so.12`, `libopenblas.so.0`, `libgomp`, `libgfortran`, `libquadmath`, `libstdc++`, `libgcc_s`, `libz` |
| `assets/maceles_off/` | `maceles_off_chno.json` (weights) + `maceles_off.cubin` (sm_80 JIT artifact) |
| `sampling/` | the whole pipeline: `make_density_scan`, `split_scan`, `les_batch`, `les_c1c2`, `scan_analysis`, `make_smoke_box` |
| `examples/smoke/` | a copy-and-run 64-water deck + `run.sh` |
| `jobs/` | `entrypoint_scan.sh` (one shard, one GPU), `q4_veMLP.yaml`, `slurm.sbatch` |

`libcublas.so.12` is **not** bundled (105 MB). `env.sh` looks for it in
`$CUDA_HOME/lib64`, `/opt/matos-tools/cuda/*/lib64`, `/usr/local/cuda*/lib64`,
or wherever the loader already finds it. On queue004 that directory is
`/opt/matos-tools/cuda/12.9/lib64`, which is picked up automatically.

## Run a scan shard

```bash
. ./env.sh
WORKDIR=/path/to/shard_1 MANIFEST=/path/to/shard_1/manifest.csv \
    jobs/entrypoint_scan.sh
```

The entry point calls `sampling/les_batch.py --run --resume`, which writes
`<WORKDIR>/samples/<sim_id>/{run/traj.dump.gz, meta.json}` and skips states that
already have a `meta.json`.

## Post-processing (CPU only, needs numpy/scipy/ase/matplotlib)

```bash
. ./env.sh
python3 sampling/scan_analysis.py --root <scan root> --out <scan root>/analysis_scan
```

## Requirements on the node

- x86-64 Linux with glibc ≥ 2.34 (CentOS 8 / Ubuntu 20.04+ / the veMLP images)
- NVIDIA driver ≥ 525 (CUDA 12 minor-version compatibility)
- one A100 (sm_80) for this build; the cubin is arch-specific
- Python 3.11 with `numpy scipy ase matplotlib` for the pipeline (LAMMPS alone needs none)

## Verify an unpacked bundle

```bash
sha256sum -c MANIFEST.sha256
ldd bin/lmp | grep -i "not found" || echo "no missing libraries"
```
