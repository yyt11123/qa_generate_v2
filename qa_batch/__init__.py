"""批量 QA 生成（10 文件）。

子模块：
- manifest    : 文件级断点 / 状态记录
- batch_runner: 主循环（扫描 inputs/batch → 复用单文件流水线）
- aggregator  : 把 output/per_file/*.xlsx 合并成 output/qa_test_all.xlsx

不影响 v3 单文件流（main.py / qa_generator/*）。
"""
