import json
import csv
import logging
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Tuple

from math_verify import parse, verify

from constants import (
    OUTPUT_DIRS,
    BENCHMARK_SIZES,
    EXTENDED_METRICS_BENCHMARKS,
    BFCL_SUBCATEGORY_MAP,
    BFCL_SUBCATEGORY_SIZES,
    METATOOL_SUBCATEGORY_SIZES,
    SUBCATEGORY_BENCHMARKS,
    EXCLUDED_MODELS,
    EXCLUDED_BENCHMARKS,
    PROJECT_ROOT,
    get_parent_category,
)
from utils import (
    extract_multiple_choice_answer,
    verify_multiple_choice,
    evaluate_bfcl_entry,
    evaluate_meta_tool_entry,
    calculate_pass_at_k,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_file_path(file_path: Path) -> Optional[Dict[str, Any]]:
    parts = file_path.parts
    try:
        output_idx = None
        for i, part in enumerate(parts):
            if part in ("output", "output_others"):
                output_idx = i
                break

        if output_idx is None:
            return None

        model = parts[output_idx + 1]
        category = parts[output_idx + 2]
        method = parts[output_idx + 3]
        remaining = parts[output_idx + 4 :]
        threshold = None

        if len(remaining) == 2:
            # Has threshold folder
            threshold_str = remaining[0]
            if threshold_str.startswith("threshold_"):
                threshold = float(threshold_str.replace("threshold_", ""))
            filename = remaining[1]
        elif len(remaining) == 1:
            filename = remaining[0]
        else:
            return None

        benchmark = filename.replace("_result.jsonl", "")

        return {
            "model": model,
            "category": category,
            "method": method,
            "threshold": threshold,
            "benchmark": benchmark,
            "file_path": file_path,
        }
    except (IndexError, ValueError) as e:
        logger.warning(f"Failed to parse path {file_path}: {e}")
        return None


def collect_all_files() -> Dict[Tuple, Path]:
    files_map = {}

    for output_dir in OUTPUT_DIRS:
        if not output_dir.exists():
            logger.warning(f"Output directory not found: {output_dir}")
            continue

        for file_path in output_dir.rglob("*_result.jsonl"):
            parsed = parse_file_path(file_path)
            if parsed is None:
                continue

            model = parsed["model"]
            method = parsed["method"]
            benchmark = parsed["benchmark"]

            if method == "oracle":
                continue
            if model in EXCLUDED_MODELS:
                continue
            if benchmark in EXCLUDED_BENCHMARKS:
                continue

            key = (model, method, parsed["threshold"], benchmark)
            files_map[key] = parsed

    return files_map


def score_entry(entry: dict, benchmark: str) -> Dict[str, Any]:
    parent_category = get_parent_category(benchmark)
    is_correct = False
    predicted = None
    ground_truth = None

    try:
        if parent_category == "general":
            ground_truth = entry.get("answer", "")
            predicted = extract_multiple_choice_answer(entry.get("response", ""))
            is_correct = verify_multiple_choice(ground_truth, predicted)
        elif parent_category == "math":
            gt_str = entry.get("answer", "")
            ground_truth = parse(f"${gt_str}$")
            predicted = parse(entry.get("response", ""))
            is_correct = verify(ground_truth, predicted)
        elif benchmark in ["bfcl-v1", "bfcl-v2"]:
            predicted, ground_truth, is_correct = evaluate_bfcl_entry(entry)
        elif benchmark == "meta-tool":
            predicted, ground_truth, is_correct = evaluate_meta_tool_entry(entry)
        else:
            logger.warning(f"Unknown benchmark: {benchmark}")
    except Exception as e:
        logger.debug(f"Error scoring entry {entry.get('id', 'unknown')}: {e}")

    return {
        "is_correct": bool(is_correct),
        "predicted": str(predicted) if predicted is not None else "",
        "ground_truth": str(ground_truth) if ground_truth is not None else "",
    }


def process_file(file_info: Dict) -> List[Dict]:
    file_path = file_info["file_path"]
    benchmark = file_info["benchmark"]
    model = file_info["model"]
    method = file_info["method"]
    threshold = file_info["threshold"]

    scored_entries = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    scoring = score_entry(entry, benchmark)

                    scored_entries.append(
                        {
                            "model": model,
                            "method": method,
                            "threshold": threshold if threshold is not None else "",
                            "benchmark": benchmark,
                            "category": file_info["category"],
                            "problem_id": entry.get("id", ""),
                            "trial": entry.get("trial", 1),
                            "sub_category": entry.get("sub_category", ""),
                            "is_correct": scoring["is_correct"],
                            "token_count": entry.get("token_length", 0),
                            "predicted": scoring["predicted"],
                            "ground_truth": scoring["ground_truth"],
                        }
                    )
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error in {file_path}: {e}")
                    continue
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")

    return scored_entries


def aggregate_and_score_all() -> List[Dict]:
    files_map = collect_all_files()
    logger.info(
        f"Found {len(files_map)} unique (model, method, threshold, benchmark) combinations"
    )

    all_scored_entries = []

    for key, file_info in files_map.items():
        logger.info(f"Processing: {key}")
        scored = process_file(file_info)
        all_scored_entries.extend(scored)
        logger.info(f"  -> {len(scored)} entries")

    return all_scored_entries


def save_to_csv(entries: List[Dict], output_path: Path):
    if not entries:
        logger.warning("No entries to save")
        return

    fieldnames = [
        "model",
        "method",
        "threshold",
        "benchmark",
        "category",
        "problem_id",
        "trial",
        "sub_category",
        "is_correct",
        "token_count",
        "predicted",
        "ground_truth",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(entries)

    logger.info(f"Saved {len(entries)} entries to {output_path}")


def calculate_metrics_for_group(entries: List[Dict], benchmark: str) -> Dict[str, Any]:
    if not entries:
        return {}

    expected_size = BENCHMARK_SIZES.get(benchmark)
    if expected_size is None:
        logger.warning(f"Unknown benchmark size for {benchmark}")
        return {}

    problems = defaultdict(list)
    for entry in entries:
        problems[entry["problem_id"]].append(entry)

    num_problems = len(problems)

    all_trials = set()
    for entries_list in problems.values():
        for e in entries_list:
            all_trials.add(e.get("trial", 1))

    max_trials = max(all_trials) if all_trials else 1

    trial_counts = defaultdict(int)
    for entry in entries:
        trial_counts[entry.get("trial", 1)] += 1

    valid_for_metrics = True
    if num_problems != expected_size:
        logger.warning(
            f"Problem count mismatch for {benchmark}: got {num_problems}, expected {expected_size}"
        )
        valid_for_metrics = False

    total_entries = len(entries)
    total_tokens = sum(e.get("token_count", 0) for e in entries)

    result = {
        "total_entries": total_entries,
        "total_problems": num_problems,
        "expected_size": expected_size,
        "avg_token_length": total_tokens / total_entries if total_entries > 0 else 0,
        "max_trials": max_trials,
        "valid_for_metrics": valid_for_metrics,
    }

    sum_accuracy = 0.0
    for problem_id, problem_entries in problems.items():
        n = len(problem_entries)
        c = sum(1 for e in problem_entries if e.get("is_correct", False))
        sum_accuracy += c / n

    result["accuracy"] = (sum_accuracy / num_problems) * 100

    if benchmark in EXTENDED_METRICS_BENCHMARKS:
        ks_to_compute = [1]
        if max_trials >= 5:
            ks_to_compute.append(5)

        pass_at_k = {}
        for k in ks_to_compute:
            pass_sum = 0.0
            for problem_id, problem_entries in problems.items():
                n = len(problem_entries)
                c = sum(1 for e in problem_entries if e.get("is_correct", False))
                pass_sum += calculate_pass_at_k(n, c, k)
            pass_at_k[k] = (pass_sum / num_problems) * 100

        result["pass@k"] = pass_at_k

        if max_trials >= 8:
            majority_correct = 0
            for problem_id, problem_entries in problems.items():
                predictions = [
                    str(e.get("predicted", ""))
                    for e in problem_entries
                    if e.get("predicted")
                ]
                if predictions:
                    counter = Counter(predictions)
                    most_common_pred = counter.most_common(1)[0][0]
                    for e in problem_entries:
                        if str(e.get("predicted", "")) == most_common_pred and e.get(
                            "is_correct", False
                        ):
                            majority_correct += 1
                            break

            result["majority_accuracy"] = (majority_correct / num_problems) * 100

            avg_at_8 = 0.0
            for problem_id, problem_entries in problems.items():
                sorted_entries = sorted(
                    problem_entries, key=lambda x: x.get("trial", 1)
                )
                first_8 = sorted_entries[:8]
                if len(first_8) == 8:
                    n_correct = sum(1 for e in first_8 if e.get("is_correct", False))
                    avg_at_8 += n_correct / 8

            result["avg@8"] = (avg_at_8 / num_problems) * 100

    return result


def calculate_metrics_for_subcategory(
    entries: List[Dict], benchmark: str, subcategory: str
) -> Dict[str, Any]:
    if not entries:
        return {}

    if benchmark == "meta-tool":
        expected_size = METATOOL_SUBCATEGORY_SIZES.get(subcategory)
    else:
        subcategory_sizes = BFCL_SUBCATEGORY_SIZES.get(benchmark, {})
        expected_size = subcategory_sizes.get(subcategory)

    if expected_size is None:
        logger.warning(f"Unknown sub-category size for {benchmark}/{subcategory}")
        return {}

    problems = defaultdict(list)
    for entry in entries:
        problems[entry["problem_id"]].append(entry)

    num_problems = len(problems)

    valid_for_metrics = num_problems == expected_size
    if not valid_for_metrics:
        logger.warning(
            f"Sub-category problem count mismatch for {benchmark}/{subcategory}: "
            f"got {num_problems}, expected {expected_size}"
        )

    total_entries = len(entries)
    total_tokens = sum(e.get("token_count", 0) for e in entries)

    result = {
        "total_entries": total_entries,
        "total_problems": num_problems,
        "expected_size": expected_size,
        "avg_token_length": total_tokens / total_entries if total_entries > 0 else 0,
        "valid_for_metrics": valid_for_metrics,
    }

    sum_accuracy = 0.0
    for problem_id, problem_entries in problems.items():
        n = len(problem_entries)
        c = sum(1 for e in problem_entries if e.get("is_correct", False))
        sum_accuracy += c / n

    result["accuracy"] = (sum_accuracy / num_problems) * 100

    return result


def calculate_all_metrics(
    entries: List[Dict], bfcl_breakdown: bool = False
) -> List[Dict]:
    groups = defaultdict(list)
    for entry in entries:
        key = (
            entry["model"],
            entry["method"],
            entry.get("threshold", ""),
            entry["benchmark"],
        )
        groups[key].append(entry)

    all_metrics = []

    for key, group_entries in groups.items():
        model, method, threshold, benchmark = key

        metrics = calculate_metrics_for_group(group_entries, benchmark)
        if not metrics:
            continue

        record = {
            "model": model,
            "method": method,
            "threshold": threshold if threshold != "" else None,
            "benchmark": benchmark,
            "sub_category": None,  # Overall benchmark metric
            **metrics,
        }
        all_metrics.append(record)

        if metrics.get("valid_for_metrics", False):
            logger.info(
                f"{model}/{method}/{threshold}/{benchmark}: "
                f"acc={metrics.get('accuracy', 0):.2f}%, "
                f"avg_tokens={metrics.get('avg_token_length', 0):.1f}"
            )

        if bfcl_breakdown and benchmark in SUBCATEGORY_BENCHMARKS:
            subcat_groups = defaultdict(list)
            for entry in group_entries:
                raw_subcat = entry.get("sub_category", "")
                if benchmark == "meta-tool":
                    normalized_subcat = raw_subcat
                else:
                    normalized_subcat = BFCL_SUBCATEGORY_MAP.get(raw_subcat, raw_subcat)
                if normalized_subcat:
                    subcat_groups[normalized_subcat].append(entry)

            for subcat, subcat_entries in subcat_groups.items():
                subcat_metrics = calculate_metrics_for_subcategory(
                    subcat_entries, benchmark, subcat
                )
                if not subcat_metrics:
                    continue

                subcat_record = {
                    "model": model,
                    "method": method,
                    "threshold": threshold if threshold != "" else None,
                    "benchmark": benchmark,
                    "sub_category": subcat,
                    **subcat_metrics,
                }
                all_metrics.append(subcat_record)

                if subcat_metrics.get("valid_for_metrics", False):
                    logger.info(
                        f"  -> {subcat}: "
                        f"acc={subcat_metrics.get('accuracy', 0):.2f}%, "
                        f"avg_tokens={subcat_metrics.get('avg_token_length', 0):.1f}"
                    )

    return all_metrics


def save_metrics_to_jsonl(metrics: List[Dict], output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        for record in metrics:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(metrics)} metric records to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate LLM response data, score, and calculate metrics."
    )
    parser.add_argument(
        "--no-bfcl-breakdown",
        action="store_true",
        default=False,
        help="Disable sub-category breakdown (enabled by default)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Prefix for output files (e.g., 'v2' -> 'v2_scored_results.csv')",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Starting data aggregation and scoring...")
    bfcl_breakdown = not args.no_bfcl_breakdown
    if bfcl_breakdown:
        logger.info("Sub-category breakdown ENABLED (default)")
    else:
        logger.info("Sub-category breakdown DISABLED")

    # Aggregate and score all entries
    all_entries = aggregate_and_score_all()
    logger.info(f"Total scored entries: {len(all_entries)}")

    # Save to CSV
    prefix = f"{args.output_prefix}_" if args.output_prefix else ""
    csv_output = PROJECT_ROOT / f"{prefix}scored_results.csv"
    save_to_csv(all_entries, csv_output)

    # Calculate metrics
    logger.info("Calculating metrics...")
    all_metrics = calculate_all_metrics(all_entries, bfcl_breakdown=bfcl_breakdown)

    # Save metrics to JSONL
    metrics_output = PROJECT_ROOT / f"{prefix}metrics_summary.jsonl"
    save_metrics_to_jsonl(all_metrics, metrics_output)

    logger.info("Done!")


if __name__ == "__main__":
    main()
