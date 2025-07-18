from transformers import TrainerCallback


class ReduceLROnPlateauCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            # print(f"[Evaluation] Current LR: {kwargs['optimizer'].param_groups[0]['lr']}")
            metric_value = metrics.get("eval_loss")
            if metric_value is not None:
                kwargs['lr_scheduler'].step(metric_value)
