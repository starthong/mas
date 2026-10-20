# 工具调用数据

## 文件

- `toolcall_sft.jsonl`：训练集，LLaMA-Factory `sharegpt` 格式。
- `dataset_info.json`：LLaMA-Factory 数据集注册。
- `samples/toolcall_sample.jsonl`：三条样例，只用于检查格式。

> 注意：`samples/toolcall_sample.jsonl` 里的 system 是简化版，只用于检查消息结构。
> 正式训练样本的 system 必须用 `src.llm.prompts.build_system_prompt(registry.describe())`
> 生成，保证训练和推理的 prompt 完全一致，否则微调效果会被 prompt 差异污染。

## 单条格式

```json
{
  "messages": [
    {"role": "system", "content": "You are a tool-calling agent. ..."},
    {"role": "user", "content": "北京今天天气怎么样？"},
    {"role": "assistant", "content": "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"北京\"}}"}
  ]
}
```

要求：

1. `system` 里放工具列表和输出协议；
2. `assistant` 的 `content` 是**纯 JSON 字符串**，不要 Markdown 围栏、不要解释文字；
3. 训练和推理使用完全相同的 system prompt；
4. 输出统一为 `{"name": "...", "arguments": {...}}`；不需要工具时用 `{"name": "final_answer", "arguments": {"answer": "..."}}`。

## 构造建议

规模 800–1500 条即可，覆盖比数量重要：

- 10–20 个工具，每个工具至少 30 条；
- 30% 多工具/并行调用；
- 15% 无需调用工具；
- 15% 缺参数、需要澄清；
- 10% 未知工具，应拒绝或改用 `final_answer`；
- 5–10% 故意带歧义、口语化、错别字。

划分训练/验证/测试时**按工具或场景划分**，不要随机切分，否则同一模板会泄漏到测试集，指标虚高。

## 快速校验

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("data/toolcall_sft.jsonl")
for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    obj = json.loads(line)
    roles = [m["role"] for m in obj["messages"]]
    assert roles[-1] == "assistant", (i, roles)
    json.loads(obj["messages"][-1]["content"])
print("ok", i)
PY
```

## 评测集

另建 `data/test/toolcall_test.jsonl`，格式相同，但保留参考答案，不要混入训练集。建议至少 200 条，并单独留 50 条**未见工具**做 OOD 测试。
