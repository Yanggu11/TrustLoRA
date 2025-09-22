from transformers import Trainer

class BatchedHypernetTrainer(Trainer):
    def __init__(self, *args, hypernet=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hypernet = hypernet

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step with hypernet precomputation,
        then fall back to the normal Trainer training step.
        """
        if self.hypernet is not None:
            self.hypernet.precompute()

        # call the parent class's training_step so it behaves exactly like Trainer
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
