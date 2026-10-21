# 工具调用数据

## 文件

- `toolcall_sft.jsonl`：训练集，LLaMA-Factory `sharegpt` 格式。
- `test/toolcall_test.jsonl`：测试集，OpenAI `messages` 格式，供 `src/eval/run_eval.py` 使用。
- `dataset_info.json`：LLaMA-Factory 数据集注册。
- `samples/toolcall_sample.jsonl`：三条训练格式样例。

> 注意：正式训练样本的 `system` 由
> `src.llm.prompts.build_system_prompt(registry.describe())` 生成，
> 保证训练和推理的 prompt 一致。

## 训练格式

```json
{
  "system": "You are a tool-calling agent. ...",
  "conversations": [
    {"from": "human", "value": "北京今天天气怎么样？"},
    {"from": "gpt", "value": "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"北京\"}}"}
  ]
}
```

`dataset_info.json` 把 `messages` 映射到 `conversations`，把 `system` 映射到 `system`：

```json
{
  "toolcall_sft": {
    "file_name": "toolcall_sft.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "system": "system"
    }
  }
}
```

## 测试格式

测试集用 OpenAI 消息格式，方便 `run_eval.py` 直接拿 `messages[:-1]` 当输入：

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

1. `gpt` / `assistant` 的内容是**纯 JSON 字符串**，不要 Markdown 围栏、不要解释；
2. 输出统一为 `{"name": "...", "arguments": {...}}`；
3. 不需要工具时用 `{"name": "final_answer", "arguments": {"answer": "..."}}`。

## 构造建议

规模 800–1500 条即可，覆盖比数量重要：

- 10–20 个工具，每个工具至少 30 条；
- 30% 多工具/并行调用；
- 15% 无需调用工具；
- 15% 缺参数、需要澄清；
- 10% 未知工具，应拒绝或改用 `final_answer`；
- 5–10% 故意带歧义、口语化、错别字。

划分训练/验证/测试时**按工具或场景划分**，不要随机切分，否则同一模板会泄漏到测试集。

## 快速校验

```bash
python - <<'PY'
import json
from pathlib import Path

for line in Path("data/toolcall_sft.jsonl").read_text(encoding="utf-8").splitlines():
    obj = json.loads(line)
    assert "system" in obj
    turns = obj["conversations"]
    assert turns[0]["from"] == "human"
    assert turns[1]["from"] == "gpt"
    json.loads(turns[1]["value"])
print("train ok")

for line in Path("data/test/toolcall_test.jsonl").read_text(encoding="utf-8").splitlines():
    obj = json.loads(line)
    assert [m["role"] for m in obj["messages"]] == ["system", "user", "assistant"]
    json.loads(obj["messages"][-1]["content"])
print("test ok")
PY
```
