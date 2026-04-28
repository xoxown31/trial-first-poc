import argparse
import json
import os
from tqdm import tqdm

from data.humaneval import load_humaneval
from methods import baseline, cot, self_repair, trial_first

METHODS = {
    "baseline": baseline.run,
    "cot": cot.run,
    "self_repair": self_repair.run,
    "trial_first": trial_first.run,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--method", choices=list(METHODS.keys()), required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    problems = load_humaneval(args.n, args.offset)
    run_fn = METHODS[args.method]
    output_path = os.path.join(args.output_dir, f"{args.method}_{args.model.replace(':', '_')}_off{args.offset}.jsonl")

    results = []
    for problem in tqdm(problems, desc=f"{args.method} | {args.model}"):
        result = run_fn(problem, args.model)
        result["task_id"] = problem["task_id"]
        results.append(result)
        with open(output_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    passed = sum(r["passed"] for r in results)
    total_tokens = sum(r["total_tokens"] for r in results)
    avg_attempts = sum(r["attempts"] for r in results) / len(results)
    avg_duration = sum(r["duration_ms"] for r in results) / len(results)

    print(f"\n{'='*50}")
    print(f"Method : {args.method}")
    print(f"Model  : {args.model}")
    print(f"pass@1 : {passed}/{len(results)} = {passed/len(results):.2%}")
    print(f"avg tokens    : {total_tokens/len(results):.0f}")
    print(f"avg attempts  : {avg_attempts:.2f}")
    print(f"avg duration  : {avg_duration:.0f} ms")
    print(f"Output : {output_path}")


if __name__ == "__main__":
    main()
