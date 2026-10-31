"""生成 v1/v2 数据配方对照所需的 seed 训练配置。"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "seeds"
OUT.mkdir(parents=True, exist_ok=True)

BASES = {
    "v1": ROOT / "configs" / "train_qlora_kaggle.yaml",
    "v2": ROOT / "configs" / "train_qlora_kaggle_v2.yaml",
}
SEEDS = (42, 43, 44, 45, 46)

for name, path in BASES.items():
    text = path.read_text(encoding="utf-8")
    for seed in SEEDS:
        body = text
        # 替换输出目录，使每个 seed 写入独立目录。
        lines = []
        for line in body.splitlines():
            if line.startswith("output_dir:"):
                lines.append(
                    f"output_dir: /kaggle/working/output/seeds/{name}-seed{seed}"
                )
            else:
                lines.append(line)
        body = "\n".join(lines) + "\n"
        if "\nseed:" not in body:
            body += f"seed: {seed}\n"
        target = OUT / f"{name}-seed{seed}.yaml"
        target.write_text(body, encoding="utf-8")
        print(target)
