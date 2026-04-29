from utils.llm_client import chat
from utils.executor import extract_code, execute_code

MAX_TRIALS = 3

SYSTEM = (
    "You are an expert Python programmer. "
    "Complete the given function. Return only the complete Python function with no explanation."
)

ANALYZE_SYSTEM = "You are a code reviewer. Be concise."

ANALYZE_TEMPLATE = """\
This Python function failed with the following error:

Code:
{code}

Error:
{error}

List the specific edge cases or implementation mistakes that likely caused this failure. \
Number each item. Be concise."""

RETRY_TEMPLATE = """\
Complete this Python function. \
Carefully address the following checklist of likely mistakes:

{checklist}

{prompt}"""


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
    code = ""
    attempt = 0

    for attempt in range(1, MAX_TRIALS + 1):
        resp = chat(model, messages)
        attempt_tokens = resp.total_tokens
        total_duration_ms += resp.duration_ms

        code = extract_code(resp.content, prompt)
        passed, error = execute_code(code, problem["test"], problem["entry_point"])

        if passed:
            tokens_per_attempt.append(attempt_tokens)
            passed_per_attempt.append(True)
            break

        # analyze failure → checklist (counts toward this attempt's tokens)
        analyze_resp = chat(
            model,
            [
                {"role": "system", "content": ANALYZE_SYSTEM},
                {"role": "user", "content": ANALYZE_TEMPLATE.format(code=code, error=error)},
            ],
        )
        attempt_tokens += analyze_resp.total_tokens
        total_duration_ms += analyze_resp.duration_ms

        tokens_per_attempt.append(attempt_tokens)
        passed_per_attempt.append(False)

        checklist = analyze_resp.content.strip()
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": RETRY_TEMPLATE.format(checklist=checklist, prompt=prompt)},
        ]

    total_tokens = sum(tokens_per_attempt)

    return {
        "method": "trial_first",
        "passed": passed,
        "error": error if not passed else "",
        "total_tokens": total_tokens,
        "tokens_per_attempt": tokens_per_attempt,
        "passed_per_attempt": passed_per_attempt,
        "duration_ms": total_duration_ms,
        "attempts": attempt,
    }
