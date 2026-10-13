"""Prompt injection 防护：可测试的规则函数，不吹成"绝对防御"。

两层：
1. 输入侧：工具返回的网页文本进 context 前包裹隔离标记
2. 输出侧：过滤/标记"忽略之前指令"类注入模式（供测试断言触发率）
"""
from __future__ import annotations

import re

UNTRUSTED_OPEN = "<untrusted_web_content>"
UNTRUSTED_CLOSE = "</untrusted_web_content>"

_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore (all |any )?(previous|prior|above) instructions",
        r"disregard (all |any )?(previous|prior|above)",
        r"忽略(之前|以上|上述|此前)(的)?(所有|全部|任何)?(的)?(指令|指示|要求|提示|规则)",
        r"forget (everything|all) (you|that)",
        r"you are now a",
        r"new system prompt",
        r"<\s*/?\s*system\s*>",
    ]
]


def wrap_untrusted(text: str) -> str:
    """网页文本进 context 前的隔离包裹。"""
    return f"{UNTRUSTED_OPEN}\n{text}\n{UNTRUSTED_CLOSE}"


def detect_injection(text: str) -> list[str]:
    """返回命中的注入模式描述列表；空列表 = 未检出。"""
    hits = []
    for pat in _INJECTION_PATTERNS:
        m = pat.search(text)
        if m:
            hits.append(m.group(0))
    return hits


def sanitize_tool_output(text: str) -> tuple[str, bool]:
    """清洗工具输出。返回 (清洗后文本, 是否检出注入)。

    策略：检出注入时移除命中行，剩余内容保留——不整段丢弃，避免误伤正常网页
    里引用攻击样本的技术文章。
    """
    if not detect_injection(text):
        return text, False
    cleaned_lines = [
        line for line in text.splitlines() if not detect_injection(line)
    ]
    return "\n".join(cleaned_lines), True
