from transformers import TrainerCallback
from scripts.training.util import extract_answer, extract_rationale

class PrintSampleCallback(TrainerCallback):
    def __init__(self, tokenizer, interval_steps=500):
        super().__init__()
        self.tokenizer = tokenizer
        self.interval_steps = interval_steps
        self.last_inputs = None  # Store the inputs of the last step
        self.last_predictions = None  # Store the predictions of the last step

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Trigger callback to print samples at specified intervals during logging.
        """
        # Trigger the callback to print samples at the specified interval
        if state.global_step % self.interval_steps == 0 and state.global_step != 0:
            if self.last_inputs is not None and self.last_predictions is not None:
                # Get the input_ids and predicted output_ids from the last batch
                input_ids = self.last_inputs['input_ids'][0]
                output_ids = self.last_predictions[0]

                # Decode input and output
                input_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

                # Extract answer and rationale from the generated output
                answer = extract_answer(output_text)
                rationale = extract_rationale(output_text)

                print(f"\nSample at step {state.global_step}:")
                print(f"Input: {input_text}")
                print(f"Predicted Answer: {answer}")
                print(f"Predicted Rationale: {rationale}\n")

    def on_step_end(self, args, state, control, **kwargs):
        """
        Capture the last batch inputs and predictions at the end of each step.
        """
        # Capture the inputs and predictions from the last batch
        self.last_inputs = kwargs.get('inputs')
        self.last_predictions = kwargs.get('predictions')