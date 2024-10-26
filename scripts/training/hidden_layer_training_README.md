
# Hidden Layer Training Guide

This README explains the flow of knowledge-distillation training, using hidden-layer from Flan-T5 base (teacher) & Flan-T5 small (student)

## Project Structure Overview
The key files used in this training are located in different directories:
- **scripts/training**: Contains `flan_training.py`, `util.py`, and `callback.py`.
- **scripts/data_preprocessing**: Contains `data_preprocessing_flan.py`.
- **config**: Contains `hyperparameters.yaml`.

The training flow is mainly handled by `flan_training.py` and involves calling a preprocessing script, training a model using knowledge distillation, and managing callbacks and utilities.

## Training Flow Explanation

### 1. Preprocessing Function (from `data_preprocessing_flan.py`)
   - **Call Preprocessing Function**: The script starts by calling the preprocessing function from `data_preprocessing_flan.py`.
   - If the **preprocessed data file does not exist**, the function download raw data from "hugging face" datasets, processes (tokenizes) it, and saves the processed data to a file for future use.
     - for tokenization, we use max_length padding to prevent losing information when tokenizes in different model
   - If the **preprocessed data file exists**, it loads the data from the file (.pkl) and proceeds directly to the next step.
   
### 2. Preprocessed Data Format
   - The **preprocessed data** is divided into training and validation datasets.
   - The format includes tokenized inputs and labels, with specific fields prepared for both training and validation.

### 3. Portioning Questions for Testing
   - The script supports using a **portion of the dataset** (e.g., 1%, 5%, etc.) to test the training process quickly.
   - The portioning logic is applied to both training and validation datasets, ensuring that the specified portion is used throughout the process.

### 4. Feeding Training Questions into the Teacher Model (Flan-T5-base)
   - The training questions are fed into the **teacher model** (`Flan-T5-base`), which generates teacher outputs.
   - These outputs are used as labels for training the student model, facilitating knowledge distillation.

### 5. Feeding Validation Questions into the Teacher Model
   - Similar to training questions, the validation questions are fed into the teacher model to generate labels.
   - The validation outputs are used to evaluate the student model’s performance during training.

### 6. Training Process (from `flan_training.py`)
   - The script initializes a **custom Trainer** that supports knowledge distillation.
   - **Loss Function**: The training process uses a combination of two loss functions:
     1. **Cross-Entropy Loss**: Standard cross-entropy loss is calculated between the student model's output and the ground truth labels (teacher outputs).
     2. **MSE Loss for Hidden States**: To implement hidden state distillation, the mean squared error (MSE) is computed between the hidden states of the teacher model (projected to match the student's dimensionality) and the student model’s hidden states.
   - **Hidden Layer Distillation**: The script extracts the last hidden layer of both the teacher and student models, applying a linear projection to match the dimensions and computing the MSE loss.
   - The total loss is a weighted sum of the cross-entropy loss and the hidden state MSE loss.
     - currently we separate weight of both loss by 50:50 ratio

### 7. Callback Mechanism (from `callback.py`)
   - The script uses a custom callback to print sample outputs during training, which helps in monitoring the training progress.
   - Other callbacks, such as early stopping, can be configured to stop the training when there is no improvement in validation performance.

### 8. Hyperparameters (from `hyperparameters.yaml`)
   - The hyperparameters are loaded from `hyperparameters.yaml`, including batch size, learning rate, number of epochs, and evaluation strategy.
   - These parameters control various aspects of the training, such as batch sizes for training and evaluation, learning rate adjustments, and the frequency of evaluations.

### 9. Evaluation Process
- After training, the evaluation process measures the accuracy of both the student model and the teacher model.
- Student Model Accuracy: The student model’s predictions on the validation dataset are compared to the ground truth labels to compute accuracy.
- Teacher Model Accuracy: The teacher model’s predictions are also evaluated on the validation dataset to serve as a baseline for the student’s performance.
- The accuracy scores are reported in the training report to understand how well the student model has learned from the teacher model.

### 10. Saving the Model and Reports
   - After training is complete, the model is saved to a specified directory.
   - A training report is generated, detailing the number of epochs, training time, final loss, and evaluation metrics.
