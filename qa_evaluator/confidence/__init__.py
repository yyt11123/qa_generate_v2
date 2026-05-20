"""RAG 置信度评测：RAGAS faithfulness 实现。

四个文件：
- log_parser.py    解析同学 RAG 系统的批量检索日志（无 API 调用）
- prompts.py       statement 抽取 + 单事实判定的 prompt 模板
- faithfulness.py  对每条 QA 跑两阶段 LLM 调用：抽事实 → 判每条事实
- confidence_runner.py  CLI 入口
"""
