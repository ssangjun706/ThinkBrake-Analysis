from pathlib import Path
import os

ROLLOUT_PREFIX = "rollout"
THINKBRAKE_PREFIX = "thinkbrake"
THINKBRAKE_PROB_PREFIX = "thinkbrake-prob"
ORACLE_PREFIX = "oracle"
THINKLESS_PREFIX = "thinkless"
NOWAIT_PREFIX = "nowait"

NOWAIT_DEFAULT_BANNED_WORDS: tuple[str, ...] = (
    "wait",
    "hmm",
    "hmmm",
    "however",
    "but",
    "alternatively",
    "check",
    "verify",
)

ROOT_DIR = Path(
    os.environ.get("THINKBRAKE_ROOT", Path(__file__).resolve().parent.parent)
)

RESULT_DIR = Path(
    os.environ.get("THINKBRAKE_OUTPUT", Path(__file__).resolve().parent.parent)
)

DATA_DIR = ROOT_DIR / THINKBRAKE_PREFIX / "data"
PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_DIRS = [
    PROJECT_ROOT / "output",
    PROJECT_ROOT / "output_others",
]

BENCHMARK_SIZES = {
    "gsm8k": 1319,
    "math500": 500,
    "aime2024": 30,
    "aime2025": 30,
    "gpqa-diamond": 198,
    "arc-challenge": 1172,
    "mmlu-redux": 5700,
    "bfcl-v1": 1150,
    "bfcl-v2": 1351,
    "meta-tool": 1492,
    "dapo-math": 300,
    "gsm8k-val": 1300,
}

PARENT_CATEGORIES = {
    "gsm8k": "math",
    "math500": "math",
    "aime2024": "math",
    "aime2025": "math",
    "gsm8k-val": "math",
    "dapo-math": "math",
    "gpqa-diamond": "general",
    "arc-challenge": "general",
    "mmlu-redux": "general",
    "bfcl-v1": "tool",
    "bfcl-v2": "tool",
    "meta-tool": "tool",
}

THRESHOLD_METHODS = {THINKBRAKE_PREFIX, THINKBRAKE_PROB_PREFIX}

ALL_METHODS = [
    ROLLOUT_PREFIX,
    THINKBRAKE_PREFIX,
    THINKBRAKE_PROB_PREFIX,
    THINKLESS_PREFIX,
    NOWAIT_PREFIX,
]

EXTENDED_METRICS_BENCHMARKS = {"math500", "aime2024", "aime2025"}
BFCL_SUBCATEGORY_MAP = {
    "simple_javascript": "simple",
    "simple_python": "simple",
    "simple_java": "simple",
    "parallel": "parallel",
    "multiple": "multiple",
    "parallel_multiple": "parallel_multiple",
    "live_simple": "simple",
    "live_parallel": "parallel",
    "live_multiple": "multiple",
    "live_parallel_multiple": "parallel_multiple",
}

BFCL_SUBCATEGORY_SIZES = {
    "bfcl-v1": {
        "simple": 550,
        "parallel": 200,
        "multiple": 200,
        "parallel_multiple": 200,
    },
    "bfcl-v2": {
        "simple": 258,
        "parallel": 16,
        "multiple": 1053,
        "parallel_multiple": 24,
    },
}

SUBCATEGORY_BENCHMARKS = {"bfcl-v1", "bfcl-v2", "meta-tool"}

METATOOL_SUBCATEGORY_SIZES = {
    "task2_subtask1": 995,
    "task2_subtask4": 497,
}

METATOOL_SUBCATEGORY_NAMES = {
    "task2_subtask1": "sub-task 1",
    "task2_subtask4": "sub-task 4",
}

EXCLUDED_MODELS = {
    "openai_gpt-oss-20b",
    "deepseek-ai_DeepSeek-R1-Distill-Llama-8B",
    "microsoft_phi-4-reasoning-plus",
}
EXCLUDED_BENCHMARKS = {"mmlu-redux", "gsm8k-val"}

MODEL_THRESHOLDS = {
    "Qwen_Qwen3-4B-Thinking-2507": 0.25,
    "Qwen_Qwen3-4B": 0.25,
    "Qwen_Qwen3-14B": 0.25,
    "Qwen_Qwen3-32B": 0.25,
    "deepseek-ai_DeepSeek-R1-Distill-Llama-8B": 1.0,
    "microsoft_phi-4-reasoning": 0.1,
}

DEFAULT_THRESHOLD = 0.25


def get_parent_category(sub_category: str) -> str:
    return PARENT_CATEGORIES.get(sub_category, "unknown")
