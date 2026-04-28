from utils.ollama_client import chat
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
Carefully address the following checklist of likely mistakes, \
then think step by step before writing the final code:

{checklist}

{prompt}"""


def run(problem: dict, model: str) -> dict:
    prompt = problem["prompt"]
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_duration_ms = 0.0

    # attempt 1: no plan, just generate
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
        total_prompt_tokens += resp.prompt_tokens
        total_completion_tokens += resp.completion_tokens
        total_duration_ms += resp.duration_ms

        code = extract_code(resp.content, prompt)
        passed, error = execute_code(code, problem["test"], problem["entry_point"])

        if passed:
            break

        # analyze failure → checklist
        analyze_resp = chat(
            model,
            [
                {"role": "system", "content": ANALYZE_SYSTEM},
                {"role": "user", "content": ANALYZE_TEMPLATE.format(code=code, error=error)},
            ],
        )
        total_prompt_tokens += analyze_resp.prompt_tokens
        total_completion_tokens += analyze_resp.completion_tokens
        total_duration_ms += analyze_resp.duration_ms

        checklist = analyze_resp.content.strip()

        # fresh context with checklist + CoT instruction
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": RETRY_TEMPLATE.format(checklist=checklist, prompt=prompt)},
        ]

    return {
        "method": "trial_first_cot",
        "passed": passed,
        "error": error if not passed else "",
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "duration_ms": total_duration_ms,
        "attempts": attempt,
    }
