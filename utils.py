import json
import re
import ast
import math
from collections import defaultdict, Counter

from typing import Union, List, Dict, Any
from constants import *


def calculate_pass_at_k(n, c, k):
    if n < k:
        return 1.0 if c > 0 else 0.0

    if c == n:
        return 1.0

    try:
        prob_fail = math.comb(n - c, k) / math.comb(n, k)
        return 1.0 - prob_fail
    except Exception:
        return 0.0


def _ast_get_name(node):
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return _ast_get_name(node.value) + "." + node.attr
    return str(node)


def _ast_resolve(node):
    if isinstance(node, ast.Call):
        func_name = _ast_get_name(node.func)
        args_dict = {}
        for keyword in node.keywords:
            args_dict[keyword.arg] = _ast_resolve(keyword.value)
        return {func_name: args_dict}
    elif isinstance(node, ast.List):
        return [_ast_resolve(e) for e in node.elts]
    elif isinstance(node, ast.Dict):
        return {
            _ast_resolve(k): _ast_resolve(v) for k, v in zip(node.keys, node.values)
        }
    elif isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _ast_resolve(node.operand)
        if isinstance(operand, (int, float)):
            return -operand

    try:
        return ast.literal_eval(node)
    except:
        return None


def qwen_parse(input_str: Union[str, List, Dict]) -> List[Dict]:
    if isinstance(input_str, (list, dict)):
        return input_str if isinstance(input_str, list) else [input_str]

    results = []
    matches = re.findall(r"<tool_call>(.*?)</tool_call>", input_str, re.DOTALL)

    if matches:
        for match in matches:
            try:
                json_str = match.strip()
                parsed = json.loads(json_str)
                results.append(parsed)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(json_str)
                    results.append(parsed)
                except:
                    pass
    else:
        try:
            parsed = json.loads(input_str)
            if isinstance(parsed, list):
                results = parsed
            elif isinstance(parsed, dict):
                results = [parsed]
        except:
            pass

    return results


def default_parse(input_str: Union[str, List, Dict]) -> List[Dict]:
    if isinstance(input_str, (list, dict)):
        return input_str if isinstance(input_str, list) else [input_str]

    input_str = input_str.strip()
    input_str = input_str.strip("`").strip()
    if input_str.startswith("json\n"):
        input_str = input_str[5:]
    elif input_str.startswith("python\n"):
        input_str = input_str[7:]

    results = []
    try:
        if not input_str.startswith("["):
            input_str = "[" + input_str
        if not input_str.endswith("]"):
            input_str = input_str + "]"

        tree = ast.parse(input_str, mode="eval")
        evaluated = _ast_resolve(tree.body)

        if isinstance(evaluated, list):
            results = evaluated
        elif isinstance(evaluated, dict):
            results = [evaluated]

    except Exception:
        try:
            cleaned_str = input_str
            if cleaned_str.startswith('["') and not cleaned_str.endswith('"]'):
                cleaned_str = cleaned_str.replace('["', "[", 1)

            tree = ast.parse(cleaned_str, mode="eval")
            evaluated = _ast_resolve(tree.body)

            if isinstance(evaluated, list):
                results = evaluated
            elif isinstance(evaluated, dict):
                results = [evaluated]
        except Exception:
            try:
                parsed = json.loads(input_str)
                if isinstance(parsed, list):
                    results = parsed
                elif isinstance(parsed, dict):
                    results = [parsed]
            except:
                pass

    final_results = []
    for item in results:
        if isinstance(item, str):
            try:
                tree = ast.parse(item, mode="eval")
                resolved = _ast_resolve(tree.body)
                if isinstance(resolved, dict):
                    final_results.append(resolved)
                else:
                    final_results.append(item)
            except:
                final_results.append(item)
        else:
            final_results.append(item)

    return final_results


def restructure_model_output(output):
    if not output:
        return output

    standardized = []
    for item in output:
        if isinstance(item, dict):
            if "name" in item and "arguments" in item and len(item) == 2:
                args = item["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        pass
                standardized.append({item["name"]: args})
            else:
                standardized.append(item)
    return standardized


def check_single_call(pred_name, pred_args, gt_item):
    if pred_name not in gt_item:
        return False

    gt_args_constraints = gt_item[pred_name]

    for arg, val in pred_args.items():
        if isinstance(val, str):
            val = val.strip()

        if arg not in gt_args_constraints:
            return False

        valid_options = gt_args_constraints[arg]
        if not isinstance(valid_options, list):
            valid_options = [valid_options]

        normalized_options = []
        for opt in valid_options:
            normalized_options.append(opt)
            if isinstance(opt, (int, float)):
                normalized_options.append(str(opt))

        if val not in valid_options and str(val) not in [str(o) for o in valid_options]:
            return False

    if len(pred_args) != len(gt_args_constraints):
        pass

    return True


def check_entry(prediction: List[Dict], ground_truth: List[Any]):
    if not prediction and not ground_truth:
        return True

    if len(prediction) != len(ground_truth):
        return False

    match_count = 0
    gt_copy = list(ground_truth)

    for pred_item in prediction:
        matched = False
        if not isinstance(pred_item, dict):
            continue

        pred_func_name = list(pred_item.keys())[0]
        pred_args = pred_item[pred_func_name]

        for idx, gt_item in enumerate(gt_copy):
            if check_single_call(pred_func_name, pred_args, gt_item):
                matched = True
                gt_copy.pop(idx)
                break

        if matched:
            match_count += 1

    return match_count == len(prediction) and len(gt_copy) == 0


def evaluate_bfcl_entry(entry: dict) -> bool:
    gt_answer = entry.get("answer")
    if gt_answer is None:
        return False, None, False

    raw_result = entry.get("response")

    if "<tool_call>" in raw_result:
        parsed = qwen_parse(raw_result)
    else:
        parsed = default_parse(raw_result)

    reconstructed = restructure_model_output(parsed)
    result = check_entry(reconstructed, gt_answer)
    return reconstructed, gt_answer, result


def extract_tool_names_from_problem(problem: str) -> List[str]:
    tool_names = []
    pattern = r"tool name:\s*([^,\n\[\]]+?)(?:,|\n|$)"
    matches = re.findall(pattern, problem, re.IGNORECASE)
    for match in matches:
        tool_name = match.strip().rstrip(",").strip()
        if tool_name:
            tool_names.append(tool_name)
    return tool_names


def extract_predicted_tools_from_response(
    response: str, tool_names: List[str], allow_multiple: bool = False
) -> List[str]:
    """
    Extract predicted tool names from response.

    Args:
        response: Model response text
        tool_names: List of valid tool names to match against
        allow_multiple: If True, extract all mentioned tools. If False, return only one.

    Returns:
        List of predicted tool names
    """
    # Pattern 1: "tool: tool_name" or "tool name: tool_name" at the end (single tool only)
    if not allow_multiple:
        tool_pattern = re.search(
            r"tool(?:\s*name)?:\s*(\S+)\s*$", response, re.IGNORECASE
        )
        if tool_pattern:
            candidate = tool_pattern.group(1).strip(".,;:\"'`*")
            if candidate.lower() == "none":
                return ["None"]
            if candidate in tool_names:
                return [candidate]

    # Pattern 2: Check for "None" or "None." as the answer
    none_pattern = re.search(r"\bNone\.?\s*$", response)
    if none_pattern:
        return ["None"]

    # Pattern 3: **bold** tool names
    bold_pattern = re.findall(r"\*\*([^*]+)\*\*", response)
    if bold_pattern:
        if allow_multiple:
            # Extract ALL matching bold tool names
            found_tools = []
            for bold_text in bold_pattern:
                bold_text = bold_text.strip()
                if bold_text in tool_names and bold_text not in found_tools:
                    found_tools.append(bold_text)
            if found_tools:
                return found_tools
        else:
            # Find the last bold tool name
            for bold_text in reversed(bold_pattern):
                bold_text = bold_text.strip()
                if bold_text in tool_names:
                    return [bold_text]

    # Pattern 4: backtick `tool_name`
    backtick_pattern = re.findall(r"`([^`]+)`", response)
    if backtick_pattern:
        if allow_multiple:
            # Extract ALL matching backtick tool names
            found_tools = []
            for bt_text in backtick_pattern:
                bt_text = bt_text.strip()
                if bt_text in tool_names and bt_text not in found_tools:
                    found_tools.append(bt_text)
            if found_tools:
                return found_tools
        else:
            # Find the last backtick tool name
            for bt_text in reversed(backtick_pattern):
                bt_text = bt_text.strip()
                if bt_text in tool_names:
                    return [bt_text]

    # Fallback: find mentioned tool names in response
    if allow_multiple:
        # Find all mentioned tool names (ordered by first appearance)
        found_tools = []
        tool_positions = []
        for tool_name in tool_names:
            pos = response.find(tool_name)
            if pos >= 0:
                tool_positions.append((pos, tool_name))
        # Sort by position and deduplicate
        tool_positions.sort(key=lambda x: x[0])
        for _, tool_name in tool_positions:
            if tool_name not in found_tools:
                found_tools.append(tool_name)
        if found_tools:
            return found_tools
    else:
        # Find the last mentioned tool name
        last_pos = -1
        last_tool = None
        for tool_name in tool_names:
            pos = response.rfind(tool_name)
            if pos > last_pos:
                last_pos = pos
                last_tool = tool_name
        if last_tool:
            return [last_tool]

    return []


def _is_multi_tool_problem(problem: str) -> bool:
    """
    Detect if the problem asks for multiple tool selection.

    Patterns indicating multi-tool:
    - "choose two appropriate tools"
    - "choose 2 tools"
    - "select two tools"
    """
    multi_patterns = [
        r"choose\s+two\s+(?:appropriate\s+)?tools",
        r"choose\s+2\s+(?:appropriate\s+)?tools",
        r"select\s+two\s+(?:appropriate\s+)?tools",
        r"select\s+2\s+(?:appropriate\s+)?tools",
        r"return\s+two\s+names?\s+of\s+the\s+tool",
        r"return\s+2\s+names?\s+of\s+the\s+tool",
    ]
    for pattern in multi_patterns:
        if re.search(pattern, problem, re.IGNORECASE):
            return True
    return False


def evaluate_meta_tool_entry(entry: dict) -> tuple:
    problem = entry.get("problem", "")
    response = entry.get("response", "")
    answer = entry.get("answer", [])

    tool_names = extract_tool_names_from_problem(problem)

    # Detect if this is a multi-tool selection problem
    allow_multiple = _is_multi_tool_problem(problem)

    predicted = extract_predicted_tools_from_response(
        response, tool_names, allow_multiple
    )

    if not isinstance(answer, list):
        answer = [answer]
    ground_truth = answer

    is_correct = set(predicted) == set(ground_truth)
    return predicted, ground_truth, is_correct


def extract_multiple_choice_answer(response: str) -> str:
    patterns = [
        r'["\*]*answer["\*]*\s*[:=]\s*["\']?([A-Da-d])["\']?',
        r"(?:the\s+)?answer\s+is[:\s]*([A-Da-d])\b",
        r"final\s+answer[:\s]*([A-Da-d])\b",
        r"(?:choice|option)[:\s]*([A-Da-d])\b",
        r"\b([A-Da-d])\s*$",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    standalone_matches = re.findall(r"\b([A-Da-d])\b", response)
    if standalone_matches:
        return standalone_matches[-1].upper()

    return ""


def verify_multiple_choice(ground_truth: str, predicted: str) -> bool:
    if not predicted:
        return False
    return ground_truth.upper().strip() == predicted.upper().strip()


def calculate_pass_at_k(n, c, k):
    if n < k:
        return 1.0 if c > 0 else 0.0

    if c == n:
        return 1.0

    try:
        prob_fail = math.comb(n - c, k) / math.comb(n, k)
        return 1.0 - prob_fail
    except Exception:
        return 0.0


def calculate_metrics(problems, total_entries=0, total_tokens=0):
    num_problems = len(problems.keys())
    if num_problems == 0:
        return {
            "total": 0,
            "correct": 0,
            "accuracy": 0.0,
            "avg_token_length": 0.0,
            "pass@k": {},
            "majority_accuracy": 0.0,
        }

    sum_accuracy = 0.0
    sum_majority = 0.0

    max_trials = 0
    for trials in problems.values():
        max_trials = max(max_trials, len(trials))

    ks_to_track = [1, 5, 8]
    ks_to_track = [k for k in ks_to_track if k <= max_trials]
    if max_trials not in ks_to_track and max_trials > 0:
        ks_to_track.append(max_trials)
    ks_to_track.sort()

    pass_at_k_sums = defaultdict(float)
    avg_at_n_sums = defaultdict(float)
    avg_at_n_counts = defaultdict(int)

    for key, trials in problems.items():
        n = len(trials)
        c = sum(1 for t in trials if t["is_correct"])

        sum_accuracy += c / n

        # 2. Majority Vote
        preds = [str(t["predicted"]) for t in trials if t["predicted"]]

        if preds:
            counter = Counter(preds)
            most_common = counter.most_common(1)
            majority_pred = most_common[0][0]

            majority_is_correct = False
            for t in trials:
                if t["predicted"] == majority_pred and t["is_correct"]:
                    majority_is_correct = True
                    break

            if majority_is_correct:
                sum_majority += 1.0

        for k in ks_to_track:
            pass_at_k_sums[k] += calculate_pass_at_k(n, c, k)

        for k in ks_to_track:
            if n >= k:
                first_k_correct = sum(1 for t in trials[:k] if t["is_correct"])
                avg_at_n_sums[k] += first_k_correct / k
                avg_at_n_counts[k] += 1

    return {
        "total": total_entries,
        "correct": sum(
            [sum(1 for t in p if t["is_correct"]) for p in problems.values()]
        ),
        "accuracy": (sum_accuracy / num_problems * 100),
        "majority_accuracy": (sum_majority / num_problems * 100),
        "avg_token_length": (
            (total_tokens / total_entries) if total_entries > 0 else 0.0
        ),
        "pass@k": {k: (pass_at_k_sums[k] / num_problems * 100) for k in ks_to_track},
        "avg@n": {
            k: (
                (avg_at_n_sums[k] / avg_at_n_counts[k] * 100)
                if avg_at_n_counts[k] > 0
                else 0.0
            )
            for k in ks_to_track
        },
    }
