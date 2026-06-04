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


if __name__ == "__main__":
    # Demo: the same problem, but three messy ways the model might answer.
    # The prompt is the function stub the model was continuing.
    prompt = 'def add(a, b):\n    """Return a + b."""\n'
    entry_point = "add"

    # Case 1: model wrapped its code in a markdown fence (strategy 1).
    completion_fenced = "Sure! Here's the solution:\n```python\ndef add(a, b):\n    return a + b\n```\nHope that helps!"

    # Case 2: model wrote only the body, continuing the prompt, then leaked a
    #         test of its own (strategy 2 — note the body needs the prompt glued on).
    completion_body = "    return a + b\n\nassert add(2, 3) == 5"

    # Case 3: model ignored the task and defined the wrong function (strategy 3).
    completion_wrong = "```python\ndef subtract(a, b):\n    return a - b\n```"

    for label, completion in [
        ("fenced block ", completion_fenced),
        ("body + leaked test", completion_body),
        ("wrong function", completion_wrong),
    ]:
        result = extract_code_with_prompt(prompt, completion, entry_point)
        print("=" * 60)
        print(f"[{label}] ->")
        print(result)

