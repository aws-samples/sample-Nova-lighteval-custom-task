import logging
import re
import numpy as np

from typing import Dict, Any
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping, MetricCategory, MetricUseCase
from lighteval.tasks.requests import Doc

from nova_lighteval_custom_task.custom_tasks.utils import create_lighteval_task


logger = logging.getLogger(__name__)

GPQA_ACCURACY = "gpqa_accuracy"
GPQA_ZS_COT = "gpqa_zs_cot"
GPQA_ZS_COT_JUDGE = "gpqa_zs_cot_judge"

def preprocess_choice(choice: str) -> str:
    if choice is None:
        return " "
    text = choice.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def preprocess(line: dict, rng: np.random.Generator) -> Dict[str, Any]:
    choices = [
        preprocess_choice(line["Correct Answer"]),
        preprocess_choice(line["Incorrect Answer 1"]),
        preprocess_choice(line["Incorrect Answer 2"]),
        preprocess_choice(line["Incorrect Answer 3"]),
    ]
    formatted_choices = [f"({lable}) {choice}" for lable, choice in zip("ABCD", choices)]
    choice_orders = list(range(4))
    rng.shuffle(choice_orders)
    formatted_choices = [formatted_choices[choice_order] for choice_order in choice_orders]
    answer = formatted_choices[choice_orders.index(0)]
    return {
        "question": line["Question"],
        "choices": formatted_choices,
        "answer": answer,
    }


def parse_answer(answer: str) -> str | None:
    # This regex matches the pattern "(A)", "(B)", "(C)", or "(D)"
    match = re.match(r"\((A|B|C|D)\)", answer)
    if match:
        return match.group(1)
    else:
        return None

def build_gpqa_zs_cot_prompt(question: str, choices: list[str]) -> str:

    formatted_choices = "\n".join(choices)

    prompt = f"""What is the correct answer to this question: {question}
    Choices:
    {formatted_choices}
    
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

def create_gpqa_zs_cot_doc(line: dict, task_name: str = "") -> Doc:
    rng = np.random.default_rng(5715)
    preprocessed_line = preprocess(line, rng)
    question = preprocessed_line["question"]
    choices = preprocessed_line["choices"]
    query = build_gpqa_zs_cot_prompt(question, choices)
    answer = parse_answer(preprocessed_line["answer"])
    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C", "D"],
        gold_index="ABCD".index(answer) if isinstance(answer, str) else answer,
        specific={"choices": choices},
    )


def create_gpqa_zs_cot_metrics() -> SampleLevelMetricGrouping:
    return SampleLevelMetricGrouping(
    metric_name=[GPQA_ACCURACY],
    higher_is_better={GPQA_ACCURACY: True},
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.SUMMARIZATION,
    sample_level_fn=GPQAZsRegexMetric().compute,
    corpus_level_fn={GPQA_ACCURACY: np.mean},
)


class GPQAZsRegexMetric:

    def compute(self, predictions: list[str], formatted_doc: Doc, **kwargs: Any) -> dict[str, float]:

        prediction = predictions[0]
        extracted_choice = self.extract_choice_with_regex(prediction)
        correct_choice = "ABCD"[formatted_doc.gold_index]

        return {GPQA_ACCURACY: 1.0 if extracted_choice == correct_choice else 0.0}

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


TASKS_TABLE = [
    create_lighteval_task(
        task_name=f"{GPQA_ZS_COT}",
        metric=[create_gpqa_zs_cot_metrics()],
        suite=["custom"],
        prompt_function=create_gpqa_zs_cot_doc,
        hf_repo="Idavidrein/gpqa",
        hf_subset="gpqa_main",
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=8192,
        stop_sequence=None,
        trust_dataset=True,
        version=0,
    )
]