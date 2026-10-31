"""检索工具使用的本地小语料。"""
from __future__ import annotations

import re

DOCUMENTS: dict[str, str] = {
    "Transformer 架构": (
        "Transformer 用自注意力建模序列，核心是 query/key/value 和多头注意力。"
        "它取消了循环结构，训练时可以并行处理整个序列。"
    ),
    "LangGraph": (
        "LangGraph 用状态图编排 LLM 应用，节点是函数，边定义控制流。"
        "它适合实现带循环、条件分支和多角色的 Agent 工作流。"
    ),
    "多智能体系统": (
        "多智能体系统让多个角色通过消息或共享状态协作。"
        "常见角色包括规划者、执行者和审查者，协作模式有串行、主管和辩论。"
    ),
    "LoRA 微调": (
        "LoRA 冻结基座权重，只训练低秩适配矩阵。"
        "它显著减少可训练参数和显存占用，适合任务格式对齐。"
    ),
    "ReAct 方法": (
        "ReAct 交替进行推理和行动：模型先写出思考，再调用工具，"
        "根据观察继续下一步，直到给出答案。"
    ),
    "RAG 检索增强": (
        "RAG 先从外部知识库检索相关片段，再把片段放进上下文让模型回答。"
        "它适合知识密集、需要引用的问答任务。"
    ),
    "vLLM 推理": (
        "vLLM 是面向大模型的高吞吐推理引擎，使用 PagedAttention 管理 KV cache，"
        "并提供 OpenAI 兼容接口。"
    ),
    "QLoRA": (
        "QLoRA 在 4-bit 量化基座上做 LoRA，进一步降低显存需求。"
        "它让 7B 级模型可以在 16GB 显存的卡上微调。"
    ),
    "工具调用": (
        "工具调用让模型输出结构化的函数名和参数，由外部程序执行。"
        "稳定性取决于 JSON 格式、参数 schema 和工具描述。"
    ),
    "JSON Schema": (
        "JSON Schema 用类型、必填字段和 additionalProperties 描述数据结构。"
        "工具调用常用它校验参数是否合法。"
    ),
    "提示工程": (
        "提示工程通过系统提示、示例和输出约束引导模型。"
        "对工具调用来说，固定输出协议和少量示例通常最有效。"
    ),
    "向量数据库": (
        "向量数据库存储文本嵌入，支持近似最近邻检索。"
        "它常用于 RAG 的语义召回，也可以和 BM25 做混合检索。"
    ),
    "BM25 检索": (
        "BM25 是经典的词频检索算法，按词项频率和文档长度归一化打分。"
        "它不需要训练，适合作为本地检索的基线。"
    ),
    "Agent 记忆": (
        "Agent 记忆分为短期上下文和长期存储。"
        "短期记忆放当前对话，长期记忆通常用向量库或结构化数据库。"
    ),
    "函数调用微调": (
        "函数调用微调用工具 schema、用户请求和目标 JSON 训练模型。"
        "数据要覆盖多工具、缺参数、无需调用和未知工具等难例。"
    ),
    "模型量化": (
        "模型量化用更低比特表示权重，常见有 8-bit、4-bit 和 GPTQ/AWQ。"
        "量化能降低显存，但可能影响生成质量和工具调用稳定性。"
    ),
    "上下文窗口": (
        "上下文窗口是模型一次能看到的 token 上限。"
        "工具结果和长期记忆都会占用窗口，过长会稀释关键信息。"
    ),
    "思维链": (
        "思维链让模型分步推理，通常能提升复杂任务表现。"
        "但在工具调用场景中，最终输出仍应是结构化 JSON。"
    ),
    "监督微调": (
        "监督微调用输入输出对训练模型，目标是学习任务格式和决策模式。"
        "数据质量和覆盖度通常比单纯增加数据量更重要。"
    ),
    "评测指标": (
        "工具调用评测可分为格式层和语义层：JSON 可解析率、schema 合规率、"
        "工具名准确率、参数匹配率和端到端任务成功率。"
    ),
}


def _tokens(text: str) -> set[str]:
    lowered = text.lower()
    latin = set(re.findall(r"[a-z0-9_]+", lowered))
    chinese = set(re.findall(r"[\u4e00-\u9fff]", lowered))
    bigrams = {
        lowered[index : index + 2]
        for index in range(len(lowered) - 1)
        if lowered[index : index + 2].strip()
    }
    return latin | chinese | bigrams


def search(query: str) -> str:
    """返回与查询最相关的本地文档。"""
    query = (query or "").strip()
    if not query:
        return "未提供检索关键词。"

    query_tokens = _tokens(query)
    scored: list[tuple[float, str, str]] = []
    for title, body in DOCUMENTS.items():
        title_tokens = _tokens(title)
        body_tokens = _tokens(body)
        score = len(query_tokens & title_tokens) * 3.0
        score += len(query_tokens & body_tokens)
        if title in query or query in title:
            score += 5.0
        if score > 0:
            scored.append((score, title, body))

    if not scored:
        return f"没有找到与“{query}”相关的本地文档。"

    scored.sort(key=lambda item: item[0], reverse=True)
    lines = []
    for rank, (_, title, body) in enumerate(scored[:3], start=1):
        lines.append(f"[{rank}] {title}: {body}")
    return "\n".join(lines)
