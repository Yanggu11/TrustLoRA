from transformers import Trainer


class BatchedHypernetTrainer(Trainer):
    def __init__(self, *args, hypernet=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hypernet = hypernet

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.hypernet is not None:
            self.hypernet.precompute(device=next(model.parameters()).device)

        return super().training_step(
            model, inputs, num_items_in_batch=num_items_in_batch
        )

    def prediction_step(
        self, model, inputs, prediction_loss_only: bool, ignore_keys=None
    ):
        if self.hypernet is not None:
            self.hypernet.precompute(device=next(model.parameters()).device)

        return super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys=ignore_keys
        )
