import logging
import re
import numpy as np

from typing import Any
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping, MetricCategory, MetricUseCase
from lighteval.tasks.requests import Doc

from nova_lighteval_custom_task.custom_tasks.utils import create_lighteval_task


logger = logging.getLogger(__name__)

MMLU_PRO_ACCURACY = "mmlu_pro_accuracy"
MMLU_PRO_ZS_COT = "mmlu_pro_zs_cot"
MMLU_PRO_ZS_COT_JUDGE = "mmlu_pro_zs_cot_judge"

def build_mmlu_pro_zs_cot_question_prompt(question: str, options: list[str]) -> str:
    formatted_options = "\n".join([f"({lable}) {option}" for lable, option in zip("ABCDEFGHIJ", options)])
    prompt = (
        f"What is the correct answer to this question: {question}\n"
        f"Options:\n{formatted_options}\n"
        f"Let's think step-by-step:"
    )
    return prompt


def build_mmlu_pro_zs_cot_prompt(question: str, options: list[str]) -> str:

    formatted_options = "\n".join([f"({lable}) {option}" for lable, option in zip("ABCDEFGHIJ", options)])

    prompt = f"""What is the correct answer to this question: {question}
    Choices:
    {formatted_options}
    
    Required format:
    - Explain your reasoning step by step
    - End with: "The correct answer is (X)." where X is A, B, C, D, E, F, G, H, I, or J
    
    Example:
    Question: What is 2 + 2?
    (A) 3 (B) 4 (C) 5 (D) 6 (E) 7 (F) 8 (G) 9 (H) 10 (I) 1 (J) 2
    
    Let me solve this step by step:
    - I need to add 2 + 2
    - 2 + 2 = 4
    - Looking at the options, 4 corresponds to choice (B)
    - Other options are incorrect
    
    The correct answer is (B).
    """

    return prompt

def create_mmlu_pro_zs_cot_doc(line: dict, task_name: str = "") -> Doc:
    question = line["question"]
    options = line["options"]
    query = build_mmlu_pro_zs_cot_prompt(question, options)
    return Doc(
        task_name=task_name,
        query=query,
        choices=options,
        gold_index=line["answer_index"],
        specific={"options": options},
    )


def create_mmlu_pro_zs_cot_metrics() -> SampleLevelMetricGrouping:
    return SampleLevelMetricGrouping(
    metric_name=[MMLU_PRO_ACCURACY],
    higher_is_better={MMLU_PRO_ACCURACY: True},
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.SUMMARIZATION,
    sample_level_fn=MmluProZsRegexMetric().compute,
    corpus_level_fn={MMLU_PRO_ACCURACY: np.mean},
)

class MmluProZsRegexMetric:

    def compute(self, predictions: list[str], formatted_doc: Doc, **kwargs: Any) -> dict[str, float]:

        prediction = predictions[0]
        extracted_choice = self.extract_choice_with_regex(prediction)
        correct_choice = "ABCDEFGHIJ"[formatted_doc.gold_index]

        return {MMLU_PRO_ACCURACY: 1.0 if extracted_choice == correct_choice else 0.0}

    def extract_choice_with_regex(self, prediction: str) -> str:
        """Extract choice using regex from model prediction"""
        prediction = prediction.replace("\n", " ").strip().lower()

        pattern_main = (
            r"the\s*correct\s*answer\s*is[:\s]*"
            r"[\(\s]*"
            r"(\\\s*[\[\(]\s*)?"
            r"\\boxed\s*\{"
            r"(\\text\s*\{)?"
            r"\s*([abcdefghij])\s*"
            r"\s*\}"
            r"(\\\s*[\]\)]\s*)?"
            r"[\)\s]*"
        )

        pattern_fallback = (
            r"the\s*correct\s*answer\s*is[:\s]*"
            r"[\(\s]*"
            r"([abcdefghij])"
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


TASKS_TABLE = [
    create_lighteval_task(
        task_name=f"{MMLU_PRO_ZS_COT}",
        metric=[create_mmlu_pro_zs_cot_metrics()],
        suite=["custom"],
        prompt_function=create_mmlu_pro_zs_cot_doc,
        hf_repo="TIGER-Lab/MMLU-Pro",
        hf_subset="default",
        hf_avail_splits=["test", "validation"],
        evaluation_splits=["test"],
        few_shots_split="dev",
        few_shots_select=None,
        generation_size=8192,
        stop_sequence=None,
        trust_dataset=True,
        version=0,
    )
]
