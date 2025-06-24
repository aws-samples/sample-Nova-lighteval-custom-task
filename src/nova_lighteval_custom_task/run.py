import os
import lighteval
import logging
from datetime import timedelta
from lighteval.models.transformers.transformers_model import TransformersModelConfig

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.endpoints.openai_model import OpenAIModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.models.model_input import GenerationParameters

accelerator = None

def setup_logging():
    """
    Configure logging settings for both console and file output.
    
    Returns:
        logger: Configured logging instance
    """
    # Create logger
    logger = logging.getLogger('lighteval')
    logger.setLevel(logging.DEBUG)
    
    # Console logging setup
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File logging setup
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler('logs/evaluation.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def main():
    """
    Main execution function for model evaluation.
    
    Configuration Guide:
    1. EvaluationTracker: Controls where and how results are saved
    2. PipelineParameters: Defines evaluation infrastructure settings
    3. Model Configuration: Specify model and generation parameters
    4. Pipeline Setup: Define which tasks to run
    """
    
    logger = setup_logging()
    logger.info("Starting model evaluation")

    evaluation_tracker = EvaluationTracker(
        output_dir="./results",
        save_details=True,
        push_to_hub=False,
        hub_results_org="test",
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        # Modify on which custom task you want to run
        custom_tasks_directory='src/nova_lighteval_custom_task/custom_tasks/mmlu/zs_cot.py'
    )

    # Configuration for OpenAI model
    generation_params = GenerationParameters(
        temperature=0,
        top_p=0.1,
    )
    
    model_config = TransformersModelConfig(
            # Modify on which model you want to run
            model_name="openai-community/gpt2",
            generation_parameters=generation_params
    )


    pipeline = Pipeline(
        # Change your preferred custom tasks here
        tasks="custom|mmlu_zs_cot:abstract_algebra|0|0",
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()

if __name__ == "__main__":
    main()