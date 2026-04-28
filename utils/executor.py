import subprocess
import tempfile
import os
import re


def extract_code(response: str, prompt: str) -> str:
    # ```python ... ``` 블록
    match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        # prompt의 def 라인이 이미 있으면 그대로, 없으면 prompt 붙임
        if code.startswith("def "):
            return code
        return prompt.rstrip() + "\n" + code

    # ``` ... ``` (언어 없는 경우)
    match = re.search(r"```\n(.*?)```", response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if code.startswith("def "):
            return code
        return prompt.rstrip() + "\n" + code

    # def로 시작하는 블록 탐색
    match = re.search(r"(def \w+.*)", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    return prompt.rstrip() + "\n    " + response.strip()


def execute_code(code: str, test: str, entry_point: str, timeout: int = 10) -> tuple[bool, str]:
    full_code = code + "\n\n" + test + f"\n\ncheck({entry_point})"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        fname = f.name

    try:
        result = subprocess.run(
            ["python", fname],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, ""
        return False, (result.stderr or result.stdout).strip()
    except subprocess.TimeoutExpired:
        return False, "TimeoutError: execution exceeded time limit"
    except Exception as e:
        return False, str(e)
    finally:
        os.unlink(fname)
