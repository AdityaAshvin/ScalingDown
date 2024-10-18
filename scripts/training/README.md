## README: Knowledge Distillation Training Script
This script performs knowledge distillation by training a student model to mimic the behavior of teacher models. The task is performed within the Mathematics domain using preprocessed AQuA-RAT data. The student model learns not only to generate correct answers but also to align with the rationale provided by teacher models.

### Arguments:
- epochs: Number of training epochs.
- batch_size: Size of each batch of training samples.
- use_gpu: Set to true to enable GPU if available, otherwise false to use CPU.
- subset_ratio: A float value indicating the fraction of the dataset to use for training (useful for testing on smaller data).

### Running the Training Script:

Run the script with the following command:
```bash
python knowledge_distillation_training.py <epochs> <batch_size> <use_gpu> <subset_ratio>
```
Example
```bash
python knowledge_distillation_training.py 10 6 true 0.5
```
This example runs the script with:
- epochs: 10 training iterations
- batch_size: 6 samples per batch
- use_gpu: true (use GPU if available)
- subset_ratio: 0.5 (use 50% of the dataset)
