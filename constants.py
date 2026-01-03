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

# Analysis project specific constants
PROJECT_ROOT = Path(__file__).resolve().parent

# Output directories (later entries override earlier ones for duplicates)
OUTPUT_DIRS = [
    PROJECT_ROOT / "output",
    PROJECT_ROOT / "output_others",  # NoWait, ThinkLess baselines
]

# Benchmark sizes (number of samples per benchmark)
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
    "gsm8k-val": 1300,
}

# Parent category mappings
PARENT_CATEGORIES = {
    "gsm8k": "math",
    "math500": "math",
    "aime2024": "math",
    "aime2025": "math",
    "gpqa-diamond": "general",
    "arc-challenge": "general",
    "mmlu-redux": "general",
    "bfcl-v1": "tool",
    "bfcl-v2": "tool",
    "meta-tool": "tool",
    "gsm8k-val": "math",
}

# Methods that require thresholds
THRESHOLD_METHODS = {THINKBRAKE_PREFIX, THINKBRAKE_PROB_PREFIX}

# All methods
ALL_METHODS = [
    ROLLOUT_PREFIX,
    THINKBRAKE_PREFIX,
    THINKBRAKE_PROB_PREFIX,
    THINKLESS_PREFIX,
    NOWAIT_PREFIX,
]

# Benchmarks supporting extended metrics (pass@k, majority, avg@n)
EXTENDED_METRICS_BENCHMARKS = {"math500", "aime2024", "aime2025"}

# BFCL sub-category normalization: maps raw sub_category to unified category
BFCL_SUBCATEGORY_MAP = {
    # bfcl-v1 sub-categories
    "simple_javascript": "simple",
    "simple_python": "simple",
    "simple_java": "simple",
    "parallel": "parallel",
    "multiple": "multiple",
    "parallel_multiple": "parallel_multiple",
    # bfcl-v2 sub-categories (live)
    "live_simple": "simple",
    "live_parallel": "parallel",
    "live_multiple": "multiple",
    "live_parallel_multiple": "parallel_multiple",
}

# BFCL sub-category sizes
BFCL_SUBCATEGORY_SIZES = {
    "bfcl-v1": {
        "simple": 550,
        "parallel": 200,
        "multiple": 200,
        "parallel_multiple": 200,
    },
    "bfcl-v2": {
        "simple": 258,  # live_simple
        "parallel": 16,  # live_parallel
        "multiple": 1053,  # live_multiple
        "parallel_multiple": 24,  # live_parallel_multiple
    },
}

# Benchmarks that support sub-category breakdown
SUBCATEGORY_BENCHMARKS = {"bfcl-v1", "bfcl-v2", "meta-tool"}

# MetaTool sub-category sizes
METATOOL_SUBCATEGORY_SIZES = {
    "task2_subtask1": 995,
    "task2_subtask4": 497,
}

# MetaTool sub-category display names
METATOOL_SUBCATEGORY_NAMES = {
    "task2_subtask1": "single",
    "task2_subtask4": "multiple",
}

# Exclusion lists - models and benchmarks to exclude from analysis
EXCLUDED_MODELS = {
    "openai_gpt-oss-20b",
    "deepseek-ai_DeepSeek-R1-Distill-Llama-8B",
}
EXCLUDED_BENCHMARKS = {
    "mmlu-redux",
}  # mmlu-redux is not used in paper tables

# Model-specific thresholds for THINKBRAKE methods
# Used in notebook to display model-appropriate threshold results
MODEL_THRESHOLDS = {
    "Qwen_Qwen3-4B-Thinking-2507": 0.25,
    "Qwen_Qwen3-4B": 0.25,
    "Qwen_Qwen3-14B": 0.25,
    "Qwen_Qwen3-32B": 0.25,
    "deepseek-ai_DeepSeek-R1-Distill-Llama-8B": 1.0,
    "microsoft_phi-4-reasoning": 0.1,
    "microsoft_phi-4-reasoning-plus: 0.1,
}

# Default threshold if model not in MODEL_THRESHOLDS
DEFAULT_THRESHOLD = 0.25


def get_parent_category(sub_category: str) -> str:
    """Get the parent category for a given sub-category/benchmark."""
    return PARENT_CATEGORIES.get(sub_category, "unknown")
