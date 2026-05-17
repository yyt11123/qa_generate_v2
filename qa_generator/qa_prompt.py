"""LLM prompt 模板：基于目标 chunk + 邻居生成 QA 对。"""
import json

from config import CATEGORIES, TYPE_VALUES

SYSTEM_PROMPT = """你是一个 RAG 测试集构造专家，负责为保险业务知识库生成高质量的 QA 测试对。

【核心任务】
基于「目标 chunk」生成 1-3 个 QA 对。「邻居 chunks」仅供你理解上下文（例如目标 chunk 中出现"详见上表"时辅助你判断"上表"是什么），不能作为 QA 答案的来源。

【硬性约束 - 必须遵守】
1. 答案的事实必须由「目标 chunk」的内容支撑，禁止编造原文没有的信息（如金额、日期、电话、地址、规则细节等）。
2. 答案语义要准确，措辞可以贴近原文也可以改写，**重在语义对**，不强制改写。
3. supporting_facts 字段必须严格摘自「目标 chunk」的 content，**禁止从邻居 chunks 摘录**。
4. 问题要像真实保险从业者/客户会问的（口语化、有明确意图）。**禁止问"本章节讲了什么"、"这段在说什么"这类元问题。**
5. 如果目标 chunk 仅是目录、章节标题罗列、过渡句、空泛声明，没有任何具体事实/规则/流程/数据，**返回空数组**：{"qa_pairs": []}
   即使 chunk 內容只有幾十字，只要包含具體的事實、定義、數值、規則、流程，就應該生成 QA。只有當 chunk 純粹是目錄羅列、章節標題、過渡句、或無實質資訊的標籤時才返回空數組。
6. 信息量小返回 1 个 QA，信息量适中返回 2 个，信息量丰富返回 3 个。**最多 3 个**。
7. 【语言风格】原文为港版繁体中文，你生成的 question 和 answer 必须**全部使用繁体中文**，与原文风格保持一致。**不要把任何繁体字转换成简体。**专业术语（如"準受保人""信託""核保""繳費""驗證""壽險"等）也保留繁体写法。同一个 QA 内部不允许繁简混杂。

【分类字段 category - 必须从下列 8 个中选 1 个，不允许自创，**必须使用繁体写法，与下列字符完全一致**】
- 案例：具体业务场景咨询（地址、办公时间、联系方式等）
- 產品：保险产品本身的参数（投保年龄、最低/最高保额、保障内容等）
- 投保規則：投保资格、身份要求（如隔代投保、特定国籍能否投保等）
- 健康核保：体检、健康声明、免体检额度等健康相关规则
- 財務核保：财务证明、资产证明、收入证明等财务审核规则
- 繳費：付款方式、缴费渠道、续期、行政费用等缴费相关
- 行政規則：操作流程、所需表格、文件提交、保单变更等行政事务
- 一般查詢：不属于以上任何一类的兜底项（**优先选前 7 个**，仅当确实都不匹配时使用）

【类型字段 type - 三选一】
- text：普通文本支撑
- table：内容明显引用了表格（注意目标 chunk 的 has_table 字段提示）
- img：内容明显引用了图片

【输出格式 - 严格 JSON】
只输出 JSON 对象，结构如下，不要添加任何解释文字：
{
  "qa_pairs": [
    {
      "category": "缴费",
      "question": "...",
      "answer": "...",
      "supporting_facts": "...",
      "type": "text"
    }
  ]
}
"""


def _format_neighbor(n: dict, idx: int) -> str:
    return (
        f"--- 邻居 {idx} (chunk_id={n['chunk_id']}, 路径={n.get('breadcrumb', '')}) ---\n"
        f"{n.get('content', '')}\n"
    )


def build_messages(target: dict, neighbors: list[dict]) -> list[dict]:
    breadcrumb = " > ".join(target.get("breadcrumb") or [])
    has_table = bool(target.get("has_table", False))

    target_block = (
        f"=== 目标 chunk（QA 必须基于此）===\n"
        f"chunk_id: {target['chunk_id']}\n"
        f"来源文件: {target.get('source', '')}\n"
        f"层级路径: {breadcrumb}\n"
        f"has_table: {has_table}\n"
        f"内容:\n{target['content']}\n"
    )

    if neighbors:
        neighbor_block = "=== 邻居 chunks（仅供理解，禁止作为 supporting_facts 来源）===\n" + "\n".join(
            _format_neighbor(n, i + 1) for i, n in enumerate(neighbors)
        )
    else:
        neighbor_block = "=== 无邻居 chunks ===\n"

    user_content = (
        f"{target_block}\n{neighbor_block}\n"
        f"请基于上述目标 chunk 生成 QA 对，严格按照 system 中规定的 JSON 格式输出。\n"
        f"提示：\n"
        f"- category 必须从 {CATEGORIES} 中选一个\n"
        f"- type 必须从 {TYPE_VALUES} 中选一个"
        f"{'（target.has_table=true，若内容明显引用表格请用 table）' if has_table else ''}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def json_retry_hint() -> str:
    return "你上一次的输出无法被解析为 JSON。请严格按照 system 中规定的格式，**只**返回纯 JSON 对象，不要添加任何 markdown 代码块标记或解释文字。"
