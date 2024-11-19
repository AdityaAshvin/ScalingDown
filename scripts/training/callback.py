# callback.py

from transformers import TrainerCallback
import torch

class LossCollectorCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.student_losses = []
        self.mse_losses = []
        self.steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Collects 'student_loss' and 'mse_loss' from logs at each logging step.
        """
        if logs is not None:
            if 'student_loss' in logs and 'mse_loss' in logs:
                self.student_losses.append(logs['student_loss'])
                self.mse_losses.append(logs['mse_loss'])
                self.steps.append(state.global_step)
            else:
                print("No 'student_loss' and 'mse_loss' found in logs.")

class EvalLossCollectorCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.eval_losses = []
        self.steps = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Collects 'eval_loss' from metrics at each evaluation step.
        """
        if metrics is not None:
            print(f"Metrics during evaluation: {metrics}")  # Debug print
            if 'eval_loss' in metrics:
                self.eval_losses.append(metrics['eval_loss'])
                self.steps.append(state.global_step)
            elif 'loss' in metrics:
                self.eval_losses.append(metrics['loss'])
                self.steps.append(state.global_step)
            else:
                print("No 'eval_loss' or 'loss' found in metrics during evaluation.")
        else:
            print("Metrics is None during evaluation.")