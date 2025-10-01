from transformers import TrainerCallback


class ReduceAlphaCallback(TrainerCallback):
    def __init__(self, alpha: float, dynamic_lora_layers: list, num_training_steps: int):
        """
        Args:
            alpha (float): Initial alpha value.
            objects (list): List of objects that have an 'alpha' attribute.
            num_training_steps (int): Total number of training steps.
        """
        self.initial_alpha = alpha
        self.dynamic_lora_layers = dynamic_lora_layers
        self.num_training_steps = num_training_steps
        self.current_step = 0
        self.alpha_decay = alpha / num_training_steps

        for layer in self.dynamic_lora_layers:
            layer.alpha = alpha

    def on_step_begin(self, args, state, control, **kwargs):
        """Reduce alpha linearly at each step."""
        if self.current_step < self.num_training_steps:
            new_alpha = max(0.0, self.initial_alpha - self.alpha_decay * self.current_step)
            for layer in self.dynamic_lora_layers:
                layer.alpha = new_alpha
            self.current_step += 1

        return control
