import logging
import numpy as np

from typing import Any
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping, MetricCategory, MetricUseCase
from lighteval.tasks.requests import Doc

from nova_lighteval_custom_task.custom_tasks.utils import create_lighteval_task
from nova_lighteval_custom_task.custom_tasks.math.utils import GetLastBoxed, MathEqual


logger = logging.getLogger(__name__)

MATH_EXACT_MATCH = "math_exact_match"
MATH_ZS_COT = "math_zs_cot"

def build_math_zs_cot_question_prompt(problem: str) -> str:
    prompt = (
        f"Solve the following math problem step by step.\n"
        f"Problem:\n{problem}\n"
        "Remember to put your answer inside \\boxed{}\n"
    )

    return prompt


def create_math_zs_cot_doc(line: dict, task_name: str = "") -> Doc:
    problem = line["problem"]
    query = build_math_zs_cot_question_prompt(problem)

    return Doc(
        task_name=task_name,
        query=query,
        choices=[],
        gold_index=[],
        specific={"target": line["solution"]},
    )


def create_math_zs_cot_metrics() -> SampleLevelMetricGrouping:
    return SampleLevelMetricGrouping(
    metric_name=[MATH_EXACT_MATCH],
    higher_is_better={MATH_EXACT_MATCH: True},
    category=MetricCategory.GENERATIVE,
	use_case=MetricUseCase.MATH,
    sample_level_fn=compute_exact_match_metric,
    corpus_level_fn={MATH_EXACT_MATCH: np.mean},
)


def compute_exact_match_metric(predictions: list[str], formatted_doc: Doc, **kwargs: Any) -> dict[str, int]:
    pred = predictions[0]
    target = formatted_doc.specific.get("target")
    processed_target = GetLastBoxed.get_last_boxed(target, "[invalid_target]")
    processed_prediction = GetLastBoxed.get_last_boxed(pred, "[invalid_pred]")
    score = int(MathEqual.grade_answer(processed_prediction, processed_target))

    return {MATH_EXACT_MATCH: score}

MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

TASKS_TABLE = [
    create_lighteval_task(
        task_name=f"{MATH_ZS_COT}:{subset}",
        metric=[create_math_zs_cot_metrics()],
        suite=["custom"],
        prompt_function=create_math_zs_cot_doc,
        hf_repo="EleutherAI/hendrycks_math",
        hf_subset=subset,
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=8192,
        stop_sequence=None,
        trust_dataset=True,
        version=0,
    )
    for subset in MATH_SUBJECTS
]
