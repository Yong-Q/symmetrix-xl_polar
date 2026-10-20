
## 15. P3 模块②落盘(2026-10-15)—— Fukui 初始化校正, v15 构建中

- **实现**(依据 extensions.py L737-766 逐行;参考核验: 两层 silu 共用同一 `nn.Activation`
  实例 → 同一常数, 引擎一致):
  - `Readout::evaluate_all`(新): 同 evaluate 链但复制 final linear 全部列(2x0e=fukui);
  - ② 块(主层循环后/readout 循环前): `sources=evaluate_all(末层 e3nn 基 feats)`
    → host 归约(2 列源和 + l=0×2 预测总量; 单图=整盒) → 0→1 钳制 →
    `factor=(target−pred)/sum`, `target=½(Q±(S−1))` → 校正核 `acc(:,0)+=src(:,0)·f_a;
    acc(:,4)+=src(:,1)·f_b` → 校正后克隆 `polar_field_independent_density`(mace L774);
  - ②b charge/spin 直传: `compute_node_energies_forces` 尾追 `total_charge=0.0/
    total_spin=1.0`(hpp 默认); 绑定 kwargs 同名默认; calculator L3394 传
    `atoms.info.get("charge"/"spin")`(mace info 口径);
  - 新访问器: `polar_fukui_sources`(N×2)、`polar_field_independent_density`(N×8)。
- **门**: `bench/polar_module2_gate.py`(已备): 引擎 fukui/fid vs
  (A) 夹具 `fukui_src` + numpy 逐式重建、(B) 活体 hooks 现取现重建; 门 5e-6。
- **风险备注**: fukui 读出头是模型中唯一的 `NonLinearBiasReadoutBlock` → has_mid
  前向路径(补丁①实现)将经受首次数值检验; 若不齐先查 silu 常数与分支次序。
