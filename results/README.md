# results

本目录保存论文中的逐样本评测记录和 summary。每个 JSON 文件都包含：

- `summary`：该配置的聚合指标；
- `records`：逐样本的 `raw`、`pred`、`ref`、`json_parse_ok`、
  `schema_ok`、`name_ok`、`exact`、`format_error` 和延迟等字段。

## 目录

| 目录 | 内容 |
|---|---|
| `kaggle-3b/` | v1 主 adapter 的 base/LoRA 格式评测、31 条系统任务、角色消融、温度稳定性和 hard/OOD/few-shot 结果 |
| `kaggle-3b-v2/` | 1,124 条扩展数据训练出的 v2 adapter 的格式评测和训练指标 |
| `seeds/` | v1/v2 各 3 个 seed 的 hard/OOD 结果 |
| `review/` | 5-seed 的 42–46 结果、55 条系统任务和 max_rounds 敏感性原始记录 |

`hard-*.json`、`ood-*.json` 和 `fewshot/*.json` 是格式评测；
`modes-*.json` 是系统任务 runner 输出；`*stability*.json` 是温度重复实验。
summary 里的数字应与报告表一致；records 是重新计算指标和失败分析的原始依据。

主 adapter 不在此目录，位于 Kaggle Dataset
`starthong/mas-3b-toolcall-output`。
