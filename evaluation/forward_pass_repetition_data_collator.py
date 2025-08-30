from transformers import Trainer
import torch

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
            
            total_loss += loss.item() / self.accumulation_steps
            self.accelerator.backward(loss / self.accumulation_steps)

        self.optimizer.step()
        self.lr_scheduler.step()
        model.zero_grad()
        
        return torch.tensor(total_loss).to(self.accelerator.device)


