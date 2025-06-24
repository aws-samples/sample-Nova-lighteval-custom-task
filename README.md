# Nova-lighteval-custom-task
This repository contains sample code demonstrating the implementation of benchmark suites used to evaluate Amazon Nova models, as detailed in the [Nova technical report](https://assets.amazon.science/96/7d/0d3e59514abf8fdcfafcdc574300/nova-tech-report-20250317-0810.pdf). The code serves as a practical example of how to generate evaluation metrics using the Amazon SageMaker Model Evaluation feature for Nova models. For more information about the evaluation framework, please refer to the [Amazon SageMaker documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/nova-model-evaluation.html).

All available benchmark have been tested on `openai-community/gpt2` within a `g5.12xlarge` instance, if you plan to load a larger model, please consider using a `p5.48xlarge` instance to avoid Out of Memory


### Benchmark List

| Benchmark | Description | Metrics | Strategy | Task Name | Task Directory |
|-----------|-------------|----------|-----------|------------|----------------|
| mmlu | Multi-task Language Understanding - Tests knowledge across 57 subjects | accuracy | zs_cot | mmlu_zs_cot | src/nova_lighteval_custom_task/custom_tasks/mmlu/zs_cot.py |
| mmlu_pro | Professional subset of MMLU - Focuses specifically on professional domains like law, medicine, accounting, and engineering | accuracy | zs_cot | mmlu_pro_zs_cot | src/nova_lighteval_custom_task/custom_tasks/mmlu_pro/zs_cot.py |
| bbh | Collection of challenging reasoning tasks designed to test advanced cognitive abilities and problem-solving skills | accuracy | fs_cot | bbh_fs_cot | src/nova_lighteval_custom_task/custom_tasks/bbh/fs_cot.py |
| gpqa | General Physics Question Answering - Specialized dataset testing understanding of physics concepts and problem-solving | accuracy | zs_cot | gpqa_zs_cot | src/nova_lighteval_custom_task/custom_tasks/gpqa/zs_cot.py |
| math | Mathematical Problem Solving - Tests mathematical reasoning abilities across various topics like algebra, calculus, and word problems | exact_match | zs_cot | math_zs_cot | src/nova_lighteval_custom_task/custom_tasks/math/zs_cot.py |

## Requirements
* (Recommended) AWS EC2 instance with Nvidia GPU 
* Python3.12 environment

## Installation
1. System Update and Dependencies

`sudo dnf update -y`

```
# Update system packages
sudo dnf update -y

# Install development tools
sudo dnf groupinstall "Development Tools" -y
sudo dnf install kernel-devel-$(uname -r) kernel-headers-$(uname -r) -y
sudo dnf install kernel-modules-extra.x86_64 -y
```

2. NVIDIA Driver Setup


```
# Create NVIDIA repository configuration
sudo tee /etc/yum.repos.d/nvidia.repo << EOF
[nvidia]
name=NVIDIA CUDA Repository
baseurl=https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64
enabled=1
gpgcheck=1
gpgkey=https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/D42D0685.pub
EOF

# Install NVIDIA drivers
sudo dnf clean all
sudo dnf -y module install nvidia-driver:latest-dkms

# Remove conflicting Java packages
sudo dnf remove java-17-amazon-corretto-devel java-21-amazon-corretto-devel
```

3. CUDA Installation

```
# Install CUDA toolkit
sudo dnf install cuda-toolkit

# Configure PATH
echo 'export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}' >> ~/.bashrc
source ~/.bashrc
```

4. Verify Installation

```
nvidia-smi
// something like
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.124.06             Driver Version: 570.124.06     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA L4                      Off |   00000000:31:00.0 Off |                    0 |
| N/A   37C    P8             12W /   72W |       1MiB /  23034MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
      
```

## Run evaluation
* Start a python virtual environment
    * `python -m venv <environment_name>`
    * `source <environment_name>/bin/activate`
* Navigate to src/nova_lighteval_custom_task and open run.py
* Modify target model and specify which custom task you want to run
* Install dependencies: under `/src` where you can find `pyproject.toml`, run `pip install .`
* Kick off a evalutaion job with `python3 src/nova_lighteval_custom_task/run.py`

##  Dataset autentication
Dataset like GPQA require you authenticate with a valid Huggingface account.
Please follow [guidance](https://huggingface.co/docs/hub/en/datasets-polars-auth) and set a HF_TOKEN:
`export HF_TOKEN="hf_xxxxxxxxxxxxx"`