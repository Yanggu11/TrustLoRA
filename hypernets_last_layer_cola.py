from datasets import load_dataset
import numpy as np
import evaluate
import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification, Trainer, TrainingArguments


# --- Custom LoRA Layer ---
class DynamicLoRALayer(nn.Module):
    def __init__(self, in_dim, r, hypernet):
        super().__init__()
        self.in_dim = in_dim
        self.r = r
        self.hypernet = hypernet
        self.training_mode = True
        self.A_fixed = None
        self.B_fixed = None

    def set_eval_mode(self):
        # Called once before evaluation
        self.training_mode = False
        with torch.no_grad():
            self.A_fixed = torch.randn(self.r, self.in_dim).to(next(self.parameters()).device)
            self.B_fixed = self.hypernet(self.A_fixed)

    def forward(self, x):
        if self.training_mode:
            A = torch.randn(self.r, self.in_dim).to(x.device)
            B = self.hypernet(A)
        else:
            A = self.A_fixed
            B = self.B_fixed

        # Apply low-rank adapter: x + (x @ B.T @ A)
        lora_out = torch.matmul(torch.matmul(x, B.T), A)
        return x + lora_out


# --- Hypernetwork: A -> B ---
class LoRAHyperNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, lora_dim):
        super().__init__()
        self.fc1 = nn.Linear(lora_dim * input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim * lora_dim)

    def forward(self, A):
        flat = A.view(-1)
        h = torch.relu(self.fc1(flat))
        B = self.fc2(h).view(A.size(1), A.size(0))  # B shape: [in_dim, r]
        return B


# --- Modified Roberta with custom LoRA ---
class RobertaWithDynamicLoRA(RobertaForSequenceClassification):
    def __init__(self, config, lora_r=8, hypernet_hidden_dim=512):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.lora_r = lora_r

        # Create hypernetwork and dynamic LoRA layer
        self.hypernet = LoRAHyperNet(self.hidden_size, hypernet_hidden_dim, lora_r)
        self.lora = DynamicLoRALayer(self.hidden_size, lora_r, self.hypernet)

        # Inject after the last layer's output dense
        self.output_dense = self.roberta.encoder.layer[-1].output.dense

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Use last hidden state from last layer for LoRA modification
        hidden_states = self.roberta.encoder.layer[-1].output.dense(outputs.hidden_states[-1])
        adapted_output = self.lora(hidden_states)

        logits = self.classifier(self.dropout(adapted_output[:, 0]))

        # Replace original logits with adapted version
        return {"logits": logits, "loss": outputs.loss} if labels is not None else {"logits": logits}



# --- Training loop (preserving your structure) ---
for it in range(3):
    print(f"====== Run {it} ===============")
    model_name = "roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    base_model = RobertaWithDynamicLoRA.from_pretrained(model_name)

    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Load and tokenize CoLA
    dataset = load_dataset("glue", "cola")

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=512)

    encoded_dataset = dataset.map(tokenize_function, batched=True)
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    metric = evaluate.load("glue", "cola")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="./outputs/roberta_base_cola",
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=1000000,
        learning_rate=4e-4,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=32,
        num_train_epochs=80,
        logging_dir="./logs/roberta_base_cola",
        metric_for_best_model="matthews_correlation",
        dataloader_num_workers=4,
        warmup_ratio=0.06,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        disable_tqdm=True,
        # remove_unused_columns=False
    )

    # Set evaluation mode hook
    def model_init():
        model = RobertaWithDynamicLoRA.from_pretrained(model_name)
        model.lora.set_eval_mode()
        return model

    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
