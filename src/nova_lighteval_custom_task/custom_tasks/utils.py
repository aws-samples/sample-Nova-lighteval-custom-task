import os
from typing import Callable, Any

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

from .constants import TMP_MODEL_PATH


def create_lighteval_task(
    task_name: str,
    metric: list[str],
    suite: list,
    prompt_function: Callable[[dict, str], Doc | None],
    hf_repo: str,
    hf_subset: str,
    hf_avail_splits: list[str],
    evaluation_splits: list[str],
    few_shots_split: str | None = None,
    few_shots_select: str | None = None,
    generation_size: int = 8192,
    stop_sequence: list[str] | None = None,
    trust_dataset: bool = True,
    version: int = 0,
) -> LightevalTaskConfig:
    """Create lighteval task.

    Arguments:
        task_name (str): Each task config should have unique task name. If having a task name like "dataset:subset" and
                         does not specify subset at evaluation config, all tasks prefixed "dataset" will be evaluated.
        suite (list[str]): Evaluation suites to which the task belongs.
        metric (list[str]): List of all the metrics for the current task.
        prompt_function (Callable[[dict, str], Doc]): Function used to create the [`Doc`] samples from each line of the evaluation dataset.
        hf_repo (str): Path of the hub dataset repository containing the evaluation information. The path will be overwritten with the local path using the load_dataset patch
        hf_subset (str): Subset used for the current task, will be default if none is selected.
        hf_avail_splits (list[str]): All the available splits in the evaluation dataset
        evaluation_splits (list[str]): List of the splits actually used for this evaluation
        few_shots_split (str): Name of the split from which to sample few-shot examples
        few_shots_select (str): Method with which to sample few-shot examples
        generation_size (int): Maximum allowed size of the generation
        stop_sequence (list[str]): Stop sequence which interrupts the generation for generative metrics.
        trust_dataset (bool): Whether to trust the dataset at execution or not
        version (int): The version of the task. Defaults to 0. Can be increased if the underlying dataset or the prompt changes.
    """
    return LightevalTaskConfig(
        name=task_name,
        suite=suite,
        prompt_function=prompt_function,
        hf_repo=hf_repo,
        hf_subset=hf_subset,
        hf_avail_splits=hf_avail_splits,
        evaluation_splits=evaluation_splits,
        few_shots_split=few_shots_split,
        few_shots_select=few_shots_select,
        generation_size=generation_size,
        metric=metric,
        stop_sequence=stop_sequence,
        trust_dataset=trust_dataset,
        version=version,
    )


def get_model_path():
    # Using the model that we that is going to be evaluated, default to TMP_MODEL_PATH if not set
    return os.environ.get('LOCAL_MODEL_PATH', TMP_MODEL_PATH)


def get_baseline_dir(base_model: str, dataset: str) -> str:
    baseline_dir = os.environ.get('BASELINE_DIR', 'baseline')
    base_model = base_model.lower()
    return os.path.join(baseline_dir, base_model, dataset)
