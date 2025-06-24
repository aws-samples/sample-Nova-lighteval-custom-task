import logging
import re
import numpy as np

from typing import Any
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping, MetricCategory, MetricUseCase
from lighteval.tasks.requests import Doc

from nova_lighteval_custom_task.custom_tasks.utils import create_lighteval_task
from nova_lighteval_custom_task.custom_tasks.bbh.constants import (
    BBH_ACCURACY,
    BBH_SUBJECTS,
    BBH_FS_COT,
    SUBSET_INSTRUCTION_MAPPING,
    SUBSET_SHOTS_MAPPING,
    SUBSET_PREAMBLE_MAPPING
)


logger = logging.getLogger(__name__)

def build_bbh_fs_cot_prompt(input: str, subset: str) -> str:

    prompt = (f"""{SUBSET_SHOTS_MAPPING[subset]}
{SUBSET_PREAMBLE_MAPPING[subset]}
{input}
{SUBSET_INSTRUCTION_MAPPING[subset]}
    """
    )
    return prompt


def create_bbh_fs_cot_doc(line: dict, task_name: str = "") -> Doc:
    subset = task_name.split(":")[1]
    query = build_bbh_fs_cot_prompt(line["input"], subset)
    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C", "D"],
        gold_index=0,
        specific={"target": line["target"]},
    )


def create_bbh_fs_cot_metrics() -> SampleLevelMetricGrouping:
    return SampleLevelMetricGrouping(
    metric_name=[BBH_ACCURACY],
    higher_is_better={BBH_ACCURACY: True},
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.SUMMARIZATION,
    sample_level_fn=BBHFsRegexMetric().compute,
    corpus_level_fn={BBH_ACCURACY: np.mean},
)

class BBHFsRegexMetric:

    def compute(self, predictions: list[str], formatted_doc: Doc, **kwargs: Any) -> dict[str, float]:

        prediction = predictions[0]
        extracted_choice = self.extract_choice_with_regex(prediction)
        correct_choice = formatted_doc.specific.get("target").lower()

        return {BBH_ACCURACY: 1.0 if extracted_choice == correct_choice else 0.0}

    def extract_choice_with_regex(self, prediction: str) -> str:
        """Extract choice using regex from model prediction by finding the last answer statement"""
        prediction = prediction.replace("\n", " ").strip().lower()

        # First, try to find the innermost answer in a nested structure
        inner_pattern = r'the\s*(?:correct\s*)?answer\s*is[:\s]*\s*(?:"|\')?(.*?)the\s*(?:correct\s*)?answer\s*is[:\s]*\s*"?(?:\*\*|\*)?([^"\.\,;*]+)'
        inner_matches = re.findall(inner_pattern, prediction, re.IGNORECASE)

        if inner_matches:
            answer = inner_matches[-1][-1].strip()
            return answer

        # If no nested structure, find regular answers
        pattern = r'the\s*(?:correct\s*)?answer\s*is[:\s]*\s*"?(?:\*\*|\*)?([^"\.\,;*]+)'
        matches = re.findall(pattern, prediction, re.IGNORECASE)

        if matches:
            answer = matches[-1].strip()
            return answer

        return "0"



TASKS_TABLE = [
    create_lighteval_task(
        task_name=f"{BBH_FS_COT}:{subset}",
        metric=[create_bbh_fs_cot_metrics()],
        suite=["custom"],
        prompt_function=create_bbh_fs_cot_doc,
        hf_repo="lukaemon/bbh",
        hf_subset=subset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=8192,
        stop_sequence=None,
        trust_dataset=True,
        version=0,
    )
    for subset in BBH_SUBJECTS
]