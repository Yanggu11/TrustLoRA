from transformers import TrainerCallback


class ReduceAlphaCallback(TrainerCallback):
    def __init__(self, alpha: float, dynamic_lora_layers: list, num_training_steps: int):
        """
        Args:
            alpha (float): Initial alpha value.
            dynamic_lora_layers (list): List of objects that have an 'alpha' attribute.
            num_training_steps (int): Total number of training steps.
        """
        self.initial_alpha = alpha
        self.dynamic_lora_layers = dynamic_lora_layers
        self.num_training_steps = num_training_steps
        self.current_step = 0
        self.alpha_decay = alpha / num_training_steps

        for layer in self.dynamic_lora_layers:
            layer.alpha = alpha

    def _set_alpha(self, new_alpha):
        """Helper method to set alpha for all dynamic LoRA layers."""
        for layer in self.dynamic_lora_layers:
            layer.alpha = new_alpha

    def _calculate_alpha(self):
        """Calculate the current alpha value based on the current step."""
        return max(0.0, self.initial_alpha - self.alpha_decay * self.current_step)

    def on_step_begin(self, args, state, control, **kwargs):
        """Reduce alpha linearly at each step."""

        if self.current_step < self.num_training_steps:
            new_alpha = self._calculate_alpha()
            self._set_alpha(new_alpha)
            self.current_step += 1

        print(f"[ReduceAlphaCallback] Begin training step={self.current_step}/{self.num_training_steps}, alpha={new_alpha:.6f}")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        """Ensure alpha is updated consistently at the end of each step."""
        if not state.is_training:
            return

        if self.current_step < self.num_training_steps:
            new_alpha = self._calculate_alpha()
            self._set_alpha(new_alpha)

        print(f"[ReduceAlphaCallback] End training step={self.current_step}/{self.num_training_steps}, alpha={new_alpha:.6f}")

    def on_evaluate(self, args, state, control, **kwargs):
        """Log alpha during evaluation without modifying it."""
        model = kwargs.get("model")
        alpha_val = None
        try:
            if model is not None and hasattr(model, "noise_alpha"):
                alpha_val = float(getattr(model, "noise_alpha"))
            elif self.dynamic_lora_layers:
                alpha_val = float(getattr(self.dynamic_lora_layers[0], "noise_alpha", "nan"))
        except Exception:
            alpha_val = "err"

        if state.is_local_process_zero:
            print(f"[ReduceAlphaCallback] Evaluation at step={int(state.global_step or 0)}, alpha={alpha_val}")