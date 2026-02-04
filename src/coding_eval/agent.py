# What this module does:
# Implement an LLM-backed agent that can write code and fix code using test feedback.
# We support A/B variants via different prompt policies.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
from llm import chat, LLMConfig
import re


VariantId = Literal["A_naive", "B_debug"]

PROMPT_VERSION = "promptB_vFinal_2026-02-01"


@dataclass
class AgentOutput:
    code: str
    response: Optional[str] = None


def _extract_code(text: str) -> str:
    """
    Robust extractor:
    - Prefer the first fenced code block (```python ... ``` or ``` ... ```).
    - If the first line inside the block is a language tag, drop it.
    - Otherwise, return the full text stripped.
    """
    t = (text or "").strip()
    if not t:
        return ""

    # Try fenced python block first
    m = re.search(r"```python\s*(.*?)```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Then any fenced block
    m = re.search(r"```\s*(.*?)```", t, flags=re.DOTALL)
    if m:
        block = m.group(1).strip()
        lines = block.splitlines()
        if lines and lines[0].strip().lower() in ("python", "py"):
            block = "\n".join(lines[1:]).strip()
        return block.strip()

    # Fallback: assume raw code
    return t



class LLMAgent:
    def __init__(self, variant: VariantId, cfg: Optional[LLMConfig] = None):
        self.variant = variant
        self.cfg = cfg or LLMConfig()

    def propose(self, signature: str, prompt: str) -> AgentOutput:
        system = (
            "You are a careful Python coding assistant.\n"
            "Use Python standard library only. Do NOT import third-party packages.\n"
            "Return ONLY valid Python code. No markdown, no explanations.\n"
            "Before emitting final code, think through which test is failing and why. Then output only code.\n"
        )

        if self.variant == "A_naive":
            user = (
                f"Task: {prompt}\n\n"
                f"Implement exactly this function signature:\n{signature}\n\n"
                "Return a complete implementation."
            )
        else:
            # B_debug: more disciplined + robust
            user = (
                f"Task: {prompt}\n\n"
                f"Implement exactly this function signature:\n{signature}\n\n"
                "Rules:\n"
                "Implement exactly the signature.\n"
                "Handle edge cases implied by the prompt.\n"
                "No helpers unless needed\n"
                "Return a single function definition\n"
            )

        txt = chat(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            cfg=self.cfg
        )
        return AgentOutput(code=_extract_code(txt), response=txt)

    def fix(
        self,
        signature: str,
        prompt: str,
        prev_code: str,
        test_output: str,
        injection: Optional[str] = None,
    ) -> AgentOutput:
        system = (
            "You are a careful Python coding assistant.\n"
            "Use Python standard library only. Do NOT import third-party packages.\n"
            "Return ONLY valid Python code. No markdown, no explanations.\n"
            "Before emitting final code, think through which test is failing and why. Then output only code.\n"
        )

        if self.variant == "A_naive":
            user = (
                f"Task: {prompt}\n\n"
                f"Signature:\n{signature}\n\n"
                f"Current code:\n{prev_code}\n\n"
                f"Pytest output:\n{test_output}\n\n"
                "Fix the code so tests pass. Return the full updated code."
            )
        else:
            user = (
                f"Task: {prompt}\n\n"
                f"Signature:\n{signature}\n\n"
                f"Current code:\n{prev_code}\n\n"
                f"Pytest output:\n{test_output}\n\n"
                "Rules:\n"
                "- If ANY test fails, you MUST change the code (no no-op outputs).\n"
                "- Make the smallest change that fixes the failing assertion(s).\n"
                "- You MAY use Python standard library (e.g., math, itertools, functools).\n"
                "- If you use any standard library module, you MUST include the needed import(s).\n"
                "- Do NOT add third-party imports.\n"
                "- Do NOT change tests.\n"
                "- Return the FULL final code (imports + function). No other text.\n"
            )

        if injection:
            user = (
                f"{user}\n\n"
                "Additional guidance (decision-time injection):\n"
                f"{injection.strip()}\n"
            )


        txt = chat(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            cfg=self.cfg
        )
        return AgentOutput(code=_extract_code(txt), response=txt)
