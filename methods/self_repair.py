from utils.llm_client import chat
from utils.executor import extract_code, execute_code

MAX_TRIALS = 3

SYSTEM = (
    "You are an expert Python programmer. "
    "Complete the given function. Return only the complete Python function with no explanation."
)

FIX_TEMPLATE = """\
Your code failed with this error:

{error}

Fix the code and return the complete corrected function."""


def run(problem: dict, model: str) -> dict:
    prompt = problem["prompt"]
    tokens_per_attempt = []
    passed_per_attempt = []
    total_duration_ms = 0.0

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Complete this Python function:\n\n{prompt}"},
    ]

    passed = False
    error = ""
    attempt = 0

    for attempt in range(1, MAX_TRIALS + 1):
        resp = chat(model, messages)
        total_duration_ms += resp.duration_ms

        code = extract_code(resp.content, prompt)
        passed, error = execute_code(code, problem["test"], problem["entry_point"])

        tokens_per_attempt.append(resp.total_tokens)
        passed_per_attempt.append(passed)

        if passed:
            break

        messages.append({"role": "assistant", "content": resp.content})
        messages.append({"role": "user", "content": FIX_TEMPLATE.format(error=error)})

    total_tokens = sum(tokens_per_attempt)

    return {
        "method": "self_repair",
        "passed": passed,
        "error": error if not passed else "",
        "total_tokens": total_tokens,
        "tokens_per_attempt": tokens_per_attempt,
        "passed_per_attempt": passed_per_attempt,
        "duration_ms": total_duration_ms,
        "attempts": attempt,
    }
