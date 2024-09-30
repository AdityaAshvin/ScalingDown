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
│   ├── lora_config.json         # Configuration file for LoRA parameters
│   └── hyperparameters.yaml     # YAML file with default hyperparameters for training
│
├── .gitignore                   # Ignore sensitive data, unnecessary system files, etc.
├── README.md                    # Overview of the repository, quick start, instructions
└── LICENSE                      # License for the project
