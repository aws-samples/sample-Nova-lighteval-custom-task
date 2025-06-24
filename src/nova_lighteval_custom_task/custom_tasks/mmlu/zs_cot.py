import logging
import re
import numpy as np

from typing import Any
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping, MetricCategory, MetricUseCase
from lighteval.tasks.requests import Doc

from nova_lighteval_custom_task.custom_tasks.utils import create_lighteval_task

logger = logging.getLogger(__name__)

# TODO: Get metric name from the recipe
MMLU_ACCURACY = "mmlu_accuracy"
MMLU_ZS_COT = "mmlu_zs_cot"
MMLU_ZS_COT_JUDGE = "mmlu_zs_cot_judge"


def build_mmlu_zs_cot_prompt(question: str, choices: list[str]) -> str:

    choices = [choices[i] for i in range(4)]

    prompt = f"""What is the correct answer to this question: {question}
    Choices:
    (A) {choices[0]}
    (B) {choices[1]}
    (C) {choices[2]}
    (D) {choices[3]}
    
    Required format:
    - Explain your reasoning step by step
    - End with: "The correct answer is (X)." where X is A, B, C, or D
    
    Example:
    Question: What is 2 + 2?
    (A) 3 (B) 4 (C) 5 (D) 6
    
    Let me solve this step by step:
    - I need to add 2 + 2
    - 2 + 2 = 4
    - Looking at the options, 4 corresponds to choice (B)
    - Options (A) 3, (C) 5, and (D) 6 are incorrect
    
    The correct answer is (B).
    """

    return prompt


def create_mmlu_zs_cot_doc(line: dict, task_name: str = "") -> Doc:
    question = line["question"]
    choices = line["choices"]
    query = build_mmlu_zs_cot_prompt(question, choices)
    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C", "D"],
        gold_index="ABCD".index(line["answer"]) if isinstance(line["answer"], str) else line["answer"],
        specific={"choices": choices},
    )


def create_mmlu_zs_cot_metrics() -> SampleLevelMetricGrouping:

    return SampleLevelMetricGrouping(
        metric_name=[MMLU_ACCURACY],
        higher_is_better={MMLU_ACCURACY: True},
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.SUMMARIZATION,
        sample_level_fn=MmluZsRegexMetric().compute,
        corpus_level_fn={MMLU_ACCURACY: np.mean},
    )


class MmluZsRegexMetric:

    def compute(self, predictions: list[str], formatted_doc: Doc, **kwargs: Any) -> dict[str, float]:

        prediction = predictions[0]
        extracted_choice = self.extract_choice_with_regex(prediction)
        correct_choice = "ABCD"[formatted_doc.gold_index]

        return {MMLU_ACCURACY: 1.0 if extracted_choice == correct_choice else 0.0}

    def extract_choice_with_regex(self, prediction: str) -> str:
        """Extract choice using regex from model prediction"""
        prediction = prediction.replace("\n", " ").strip().lower()

        pattern_main = (
            r"the\s*correct\s*answer\s*is[:\s]*"
            r"[\(\s]*"
            r"(\\\s*[\[\(]\s*)?"
            r"\\boxed\s*\{"
            r"(\\text\s*\{)?"
            r"\s*([abcd])\s*"
            r"\s*\}"
            r"(\\\s*[\]\)]\s*)?"
            r"[\)\s]*"
        )

        pattern_fallback = (
            r"the\s*correct\s*answer\s*is[:\s]*"
            r"[\(\s]*"
            r"([abcd])"
            r"[\)\s]*"
        )

        match = re.search(pattern_main, prediction, re.IGNORECASE)

        if match:
            pred_choice = match.group(3).upper()
        else:
            match = re.search(pattern_fallback, prediction, re.IGNORECASE)
            if match:
                pred_choice = match.group(1).upper()
            else:
                # No match at all
                return "0"
        return pred_choice


MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions"
]

TASKS_TABLE = [
    create_lighteval_task(
        task_name=f"{MMLU_ZS_COT}:{subset}",
        metric=[create_mmlu_zs_cot_metrics()],
        suite=["custom"],
        prompt_function=create_mmlu_zs_cot_doc,
        hf_repo="lighteval/mmlu",
        hf_subset=subset,
        hf_avail_splits=["auxiliary_train", "test", "validation", "dev"],
        evaluation_splits=["test"],
        few_shots_split="dev",
        few_shots_select=None,
        generation_size=8192,
        stop_sequence=None,
        trust_dataset=True,
        version=0,
    )
    for subset in MMLU_SUBJECTS
]