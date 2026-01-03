import logging
import argparse
import json
from collections import defaultdict
import re
import math

from math_verify import parse, verify
from fractions import Fraction
from pathlib import Path

from constants import (
    RESULT_DIR,
    THINKBRAKE_PREFIX,
    THINKBRAKE_PROB_PREFIX,
    THINKLESS_PREFIX,
    NOWAIT_PREFIX,
    ROLLOUT_PREFIX,
    get_parent_category,
)

from utils import (
    extract_multiple_choice_answer,
    verify_multiple_choice,
    evaluate_bfcl_entry,
    evaluate_meta_tool_entry,
    calculate_metrics,
)


PREFIX_MAP = {
    "thinkbrake": THINKBRAKE_PREFIX,
    "thinkbrake-prob": THINKBRAKE_PROB_PREFIX,
    "thinkless": THINKLESS_PREFIX,
    "nowait": NOWAIT_PREFIX,
    "rollout": ROLLOUT_PREFIX,
}

# Modes that have threshold subfolders (threshold_X directories)
THRESHOLD_MODES = {THINKBRAKE_PREFIX, THINKBRAKE_PROB_PREFIX}


# --- Parsing Logic from minjaeoh/fpa/HALOs/train/math_parsingutil.py ---


def normalize_number_format(string):
    """쉼표 제거 및 `th` 접미사 처리"""
    if string is None:
        return None
    string = string.replace(",", "")  # 쉼표 제거
    string = re.sub(r"(\d+)th", r"\1", string)  # "12th" → "12"
    return string.strip()


def convert_decimal_to_fraction(string):
    """소수를 분수로 변환"""
    try:
        if string is None:
            return None

        value = float(string)

        # 무한대 또는 NaN 값 방지
        if not math.isfinite(value):
            return string

        fraction = Fraction(value).limit_denominator(
            1000
        )  # 1000까지 제한하여 근사값 방지
        return f"\\frac{{{fraction.numerator}}}{{{fraction.denominator}}}"
    except ValueError:
        return string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = _remove_text_env(string)
    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _remove_text_env(string):
    """LaTeX의 \text{} 환경을 제거하고 내부 문자열만 남김"""
    return re.sub(r"\\text\{(.*?)\}", r"\1", string)


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        # print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        # pdb.set_trace()
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def check_equivalence(answer, ground_truth):
    """답변이 정답과 동등한지 확인"""
    if answer == "NO_ANSWER" or answer is None:
        return False

    if ground_truth is None:
        # print("Warning: ground_truth is None")
        return False

    # 1️⃣ 숫자 형식 정규화
    normalized_answer = normalize_number_format(answer)
    normalized_ground_truth = normalize_number_format(ground_truth)

    # 2️⃣ 분수 & 소수 변환
    converted_answer = convert_decimal_to_fraction(normalized_answer)
    converted_ground_truth = convert_decimal_to_fraction(normalized_ground_truth)

    # 3️⃣ 기존 math_parsingutil.is_equiv() 호출
    return is_equiv(normalized_answer, normalized_ground_truth) or is_equiv(
        converted_answer, converted_ground_truth
    )


def extract_boxed(text: str):
    """
    ▸ \boxed{…}  또는  \boxed …  표현의 **가장 마지막** 내용을 추출
    ▸ 없으면 None 반환
    """
    if not text:
        return None
    # ①  마지막 `\boxed{…}` 찾기
    idx = text.rfind(r"\boxed{")
    if idx != -1:
        idx += len(r"\boxed{")
        depth = 1  # 중괄호 깊이
        i = idx
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:  # 짝이 맞았을 때
            return text[idx : i - 1].strip()
    # ②  마지막 `\boxed ...`  (중괄호 없는 형태) 찾기
    #    예:  \boxed 5   ,  \boxed -\frac12
    pattern = r"\\boxed\s+([^\s\\\$\}\.,]+)"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def extract_answer(text):
    if not text:
        return None
    boxed_content = extract_boxed(text)
    if boxed_content:
        ans = boxed_content.strip()
        if ans == "X":
            return "NO_ANSWER"
        return ans
    final_pattern = r"(?i)final answer is[:\s]+\$?(.+?)(?:\$|\n|$)"
    matches = re.findall(final_pattern, text)
    if matches:
        ans = matches[-1].strip()
        if ans == "X":
            return "NO_ANSWER"
        return ans
    ans_pattern = r"(?i)answer[:\s]+\$?(.+?)(?:\$|\n|$)"
    matches = re.findall(ans_pattern, text)
    if matches:
        ans = matches[-1].strip()
        if ans == "X":
            return "NO_ANSWER"
        return ans
    return None


# ---------------------------------------------------------------------


def _evaluate_jsonl_file(file_path: str, sub_category: str) -> dict:
    problems = defaultdict(list)

    total_tokens = 0
    total_entries = 0
    parent_category = get_parent_category(sub_category)

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            for line in f:
                item = json.loads(line.strip())
                total_entries += 1
                total_tokens += item.get("token_length", 0)

                is_correct = False
                predicted = None
                ground_truth = None

                if parent_category == "general":
                    ground_truth = item["answer"]
                    predicted = extract_multiple_choice_answer(item["response"])
                    is_correct = verify_multiple_choice(ground_truth, predicted)
                elif parent_category == "math":
                    # Use migrated parsing logic
                    predicted = extract_answer(item["response"])
                    ground_truth = item["answer"]
                    is_correct_loc = check_equivalence(predicted, ground_truth)

                    pred_lib = parse(item["response"])
                    gt_lib = parse(f"${item['answer']}$")
                    is_correct_lib = verify(gt_lib, pred_lib)
                    is_correct = is_correct_loc or is_correct_lib

                elif sub_category in ["bfcl-v1", "bfcl-v2"]:
                    predicted, ground_truth, is_correct = evaluate_bfcl_entry(item)
                elif sub_category == "meta-tool":
                    predicted, ground_truth, is_correct = evaluate_meta_tool_entry(item)
                else:
                    raise ValueError(
                        f"Unknown parent category found: {parent_category}"
                    )

                problems[item["id"]].append(
                    {
                        "predicted": str(predicted),
                        "is_correct": is_correct,
                        "ground_truth": str(ground_truth),
                    }
                )

        except Exception as e:
            logging.error(f"Error evaluating the file {file_path}: {e}")
            return None

    return calculate_metrics(
        problems,
        total_entries=total_entries,
        total_tokens=total_tokens,
    )
