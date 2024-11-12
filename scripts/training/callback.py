# callback.py
from transformers import TrainerCallback
from scripts.training.util import extract_answer
import torch

class PrintSampleCallback(TrainerCallback):
    def __init__(self, tokenizer, sample_input_text, correct_answer, interval_steps=100):
        super().__init__()
        self.tokenizer = tokenizer
        self.sample_input_text = sample_input_text
        self.correct_answer = correct_answer
        self.interval_steps = interval_steps
        self.sample_input_ids = self.tokenizer.encode(self.sample_input_text, return_tensors='pt')

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        """
        Trigger callback to print samples at specified intervals during logging.
        """
        # Trigger the callback to print samples at the specified interval
        if state.global_step % self.interval_steps == 0 and state.global_step != 0:
            if model is not None:
                model.eval()
                with torch.no_grad():
                    input_ids = self.sample_input_ids.to(model.device)
                    output_ids = model.generate(
                        input_ids=input_ids,
                        max_length=150,
                        num_beams=4,
                        early_stopping=True
                    )
                output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

                # Extract the predicted answer from the generated output
                predicted_answer = extract_answer(output_text)

                print(f"\nSample at step {state.global_step}:")
                print(f"Input: {self.sample_input_text}")
                print(f"Correct Answer: {self.correct_answer}")
                print(f"Predicted Answer: {predicted_answer}")
                print(f"Model Output: {output_text}\n")

        super().on_log(args, state, control, logs=logs, **kwargs)

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