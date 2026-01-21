import torch
from transformers import Trainer


class SimpleGradientAccumulationTrainer(Trainer):
    def __init__(self, *args, accumulation_steps=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.accumulation_steps = accumulation_steps

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform training step with gradient accumulation on the same batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        total_loss = 0.0
        model.zero_grad()

        for i in range(self.accumulation_steps):
            with self.autocast_smart_context_manager():
                outputs = model(**inputs)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            # Scale loss for accumulation
            scaled_loss = loss / self.accumulation_steps
            self.accelerator.backward(scaled_loss)
            total_loss += loss.item()

        self.optimizer.step()
        self.lr_scheduler.step()
        model.zero_grad()

        return torch.tensor(total_loss).to(self.accelerator.device)


# ! please don't remove comment below; I will do this when the time comes

# class SimpleGradientAccumulationTrainer(Trainer):
#     def __init__(self, *args, accumulation_steps=10, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.accumulation_steps = accumulation_steps

#     def training_step(self, model, inputs, num_items_in_batch=None):
#         """
#         Perform training step with gradient accumulation on the same batch.
#         """
#         model.train()
#         inputs = self._prepare_inputs(inputs)

#         total_loss = 0.0
#         model.zero_grad()

#         # Accumulate gradients
#         accumulated_so_far = {}

#         for i in range(self.accumulation_steps):
#             with self.autocast_smart_context_manager():
#                 outputs = model(**inputs)
#                 loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

#             # Scale loss for accumulation
#             scaled_loss = loss / self.accumulation_steps
#             self.accelerator.backward(scaled_loss)

#             # 🔎 Inspect gradients during this backward pass
#             for name, param in model.named_parameters():
#                 if "hypernet.fc1.weight" in name and param.grad is not None:
#                     # Clone current grad (this is accumulated already, so subtract previous snapshot if you want only "delta")
#                     grad_now = param.grad.detach().clone().cpu()

#                     # To approximate "just this backward pass", subtract grad before backward
#                     if i == 0:
#                         grad_delta = grad_now.clone()
#                     else:
#                         grad_delta = grad_now - accumulated_so_far.get("name", 0.0)

#                     print(f"[Step {i+1}] {name} | Grad delta mean: {grad_delta.abs().mean():.6f} | Norm: {grad_delta.norm():.6f}")

#                     # Save snapshot for next step comparison
#                     accumulated_so_far[name] = grad_now.clone()

#         print("============================")

#         self.optimizer.step()
#         self.lr_scheduler.step()
#         model.zero_grad()

#         return torch.tensor(total_loss).to(self.accelerator.device)
