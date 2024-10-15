# ScalingDown
The project is about determining how small a Large Language Model (LLM) can be while still outperforming a larger model on a specific task, using multiple larger models to train. We’ll use the TinyLLM framework to implement student-teacher models, testing the trade-offs between size, performance, and efficiency

## Project Structure (Draft)
```text
project-root/
│
├── data/
│   ├── AQua/                    # AQUA-RAT (Algebra Question Answering with Rationales) Dataset
│   └── GSM8K/                   # Preprocessed and formatted datasets for modeling
│
├── scripts/
│   ├── data_preprocessing/      # Scripts for cleaning, formatting, and batching datasets
│   ├── teacher_llm/             # Scripts for sending batches to teacher LLMs and collecting data
│   ├── training/                # Training scripts for student models, including LoRA implementation
│   ├── evaluation/              # Evaluation scripts to measure model performance against benchmarks
│   └── environment/             # Environment setup scripts, requirements.txt, and setup.sh
│
├── models/
│   ├── checkpoints/             # Saved model checkpoints during training
│   ├── student_model/           # Student model versions (pre-trained, fine-tuned)
│   └── README.md                # Documentation about model versions and checkpoints
│
├── results/
│   ├── training_logs/           # Logs for training runs, hyperparameter tuning, etc.
│   ├── evaluation_reports/      # Reports of evaluations, performance comparisons
│   └── README.md                # Summary of results and where to find specific logs
│
├── docs/                              # General documentation structure
│   ├── approach_and_methodology.md    # Description of the project, including goals and methods
│   ├── findings_and_results.md        # Findings, results, analysis, and visualizations
│   └── presentation/                  # Materials for the final presentation (slides, etc.)
│
├── config/
│   ├── carc_setup.sh            # Script for setting up the environment on CARC
│   ├── environment.yaml         # YAML file with Conda environment variables
│   ├── hyperparameters.yaml     # YAML file with default hyperparameters for training
|   └── lora_config.json         # Configuration file for LoRA parameters
│
├── .gitignore                   # Ignore sensitive data, unnecessary system files, etc.
├── README.md                    # Overview of the repository, quick start, instructions
└── LICENSE                      # License for the project
```

## Installing Conda

After cloning in the repository, create the Conda environment using:
```shell
conda env create -f /<path to SCALINGDOWN repo>/config/environment.yml
```
Then activate the environment using:
```shell
conda activate scalingDownEnv
```
At this point the environment should be set up with the following packages:
```text
cudatoolkit (version 11.8)
cudnn (version 8.9.2.26)
matplotlib
numpy
pandas
tensorflow
```

To add more packages into the environment, use (numpy and tensorflow used as examples):
```shell
conda install numpy tensorflow
```

To remove packages from the environment, use:

```shell
conda remove numpy
```

Once you have added or removed packages from the environment, use:

```shell
conda env export > <path to where you want the yaml file>/environment.yml
```

Update the environment (removing build), use:
```shell
conda env export --no-builds > environment.yml
```

## Path Variables
We'll need to figure out if the project will rely on specific environment variables, and if so, set them
manually. This might be a better option than hardcoding them into the environment.yml file. 

## GPU Support
It will be important to ensure that TensorFlow is set up for GPU support, since CARC has GPUs available. 
We can verify this with the following
```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```
## Downloading Datasets
Here's some example code for later for downloading datasets in Python:
```python
import requests

url = "https://example.com/dataset.zip"
response = requests.get(url)

with open("dataset.zip", "wb") as file:
    file.write(response.content)
```
