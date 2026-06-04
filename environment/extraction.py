"""Code extraction from model completions.

The article uses a regex-based extractor to pull a runnable function out of
`prompt + completion`. The model may wrap code in markdown fences or just emit a
function body, so we try a few patterns and fall back to stitching the prompt
(the function signature) onto the completion.
"""

import re


def extract_code_with_prompt(prompt: str, completion: str, entry_point: str):
    """Extract runnable code that defines `entry_point`.

    Returns the code string, or None if no definition of `entry_point` is found.
    """
    text = prompt + completion

    # 1) Prefer fenced code blocks if present.
    fenced = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    for block in fenced:
        if f"def {entry_point}" in block:
            return block

    # 2) Otherwise, if the combined text already defines the function, use it
    #    up to the start of the test harness (if any leaked in).
    if f"def {entry_point}" in text:
        # Cut at an obvious test marker so executor's own test is the only one.
        for marker in ("\ndef check(", "\n# Test", "\nassert "):
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx]
        return text

    # 3) Nothing usable.
    return None
