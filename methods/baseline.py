from utils.llm_client import chat
from utils.executor import extract_code, execute_code

SYSTEM = (
    "You are an expert Python programmer. "
    "Complete the given function. Return only the complete Python function with no explanation."
)


def run(problem: dict, model: str) -> dict:
    prompt = problem["prompt"]

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Complete this Python function:\n\n{prompt}"},
    ]

    resp = chat(model, messages)
    code = extract_code(resp.content, prompt)
    passed, error = execute_code(code, problem["test"], problem["entry_point"])

    return {
        "method": "baseline",
        "passed": passed,
        "error": error,
        "prompt_tokens": resp.prompt_tokens,
        "completion_tokens": resp.completion_tokens,
        "total_tokens": resp.total_tokens,
        "duration_ms": resp.duration_ms,
        "attempts": 1,
    }
