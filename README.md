# MPM_codex — Phase 2

该目录是在 phase-1 基线迁移之上的二阶段工程化重构。核心目标是把找矿预测实验拆成可注册、可配置、可验证的组件，而不是继续在 `Experiment` 中增加算法分支。

主要入口：

```bash
# 查看组件
python scripts/list_components.py

# 只验证配置
python run.py --config configs/experiments/lachlan_rf_phase2.yaml --validate-only

# 运行实验
python run.py --config configs/experiments/lachlan_rf_phase2.yaml

# 配置 -> shell
python scripts/render_run_sh.py --config configs/experiments/lachlan_rf_phase2.yaml

# 单元/集成测试
pytest -q
```

架构说明：[`docs/PHASE2_ARCHITECTURE_CN.md`](docs/PHASE2_ARCHITECTURE_CN.md)

使用说明：[`docs/PHASE2_USAGE_CN.md`](docs/PHASE2_USAGE_CN.md)

Phase-1 迁移背景仍保留在 [`docs/PHASE1_MIGRATION.md`](docs/PHASE1_MIGRATION.md)。旧实验 YAML 可继续加载，配置层会自动迁移为 phase-2 `ExperimentSpec`。

注意：`deep_edge_prediction` 当前只是显式能力占位；空间 CV、真实地质知识/谓词和更多模型需要按注册表契约继续实现，不能视为已经具备。
