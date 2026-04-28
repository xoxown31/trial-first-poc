from utils.ollama_client import chat
from utils.executor import extract_code, execute_code

SYSTEM = (
    "You are an expert Python programmer. "
    "Think step by step before writing code. "
    "After reasoning, return the complete Python function."
)

USER_TEMPLATE = """\
Complete this Python function. Think step by step, then write the complete function.

{prompt}"""


def run(problem: dict, model: str) -> dict:
    prompt = problem["prompt"]

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TEMPLATE.format(prompt=prompt)},
    ]

    resp = chat(model, messages)
    code = extract_code(resp.content, prompt)
    passed, error = execute_code(code, problem["test"], problem["entry_point"])

    return {
        "method": "cot",
        "passed": passed,
        "error": error,
        "prompt_tokens": resp.prompt_tokens,
        "completion_tokens": resp.completion_tokens,
        "total_tokens": resp.total_tokens,
        "duration_ms": resp.duration_ms,
        "attempts": 1,
    }
