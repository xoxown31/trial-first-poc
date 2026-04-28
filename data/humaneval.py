from human_eval.data import read_problems


def load_humaneval(n: int = None, offset: int = 0):
    problems = list(read_problems().values())
    problems = problems[offset:]
    if n is not None:
        problems = problems[:n]
    return problems
