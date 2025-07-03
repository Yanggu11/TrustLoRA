# %%
from datasets import load_dataset
import numpy as np
import evaluate
import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device} on Hypernet with RoBERTa on CoLA")

model_name = "roberta-base"

# %%
tokenizer = RobertaTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=512)

dataset = load_dataset("glue", "cola")

encoded_dataset = dataset.map(tokenize_function, batched=True)
encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

metric = evaluate.load("glue", "cola")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# %%
# --- Hypernetwork: A -> B ---
class LoRAHyperNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, lora_dim):
        super().__init__()
        self.fc1 = nn.Linear(lora_dim * input_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim * 2 * lora_dim)

    def forward(self, inpt):
        flat = inpt.view(-1)
        h = torch.relu(self.fc1(flat))
        opt = self.fc2(h).view(inpt.shape)  # B shape: [in_dim, r]

        return (opt[0], opt[1].T)

# %%
import gc

# full dynamic layer acting as LoRA path: lora_B(lora_A(x)) ≈ DynamicLoRALayer(x)
class DynamicLoRALayer(nn.Module):
    def __init__(self, hidden_size, r, hypernet: nn.Module):
        super().__init__()
        self.hidden_size = hidden_size
        self.r = r
        self.hypernet = hypernet

        self.weight = torch.tensor(0.0, dtype=self.hypernet.fc1.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            inpt = torch.randn((2, self.r, self.hidden_size)).to(x.device)

            A, B = self.hypernet(inpt)  # A: [r, hidden], B: [hidden, r]
            output = torch.matmul(torch.matmul(x, B), A)  # returns [batch, seq_len, hidden]
        else:
            with torch.no_grad():
                gc.collect()
                inpt = torch.randn((2, self.r, self.hidden_size)).to(x.device)

                A, B = self.hypernet(inpt)  # A: [r, hidden], B: [hidden, r]
                output = torch.matmul(torch.matmul(x, B), A)  # returns [batch, seq_len, hidden]
        return output

# %%
class RobertaWithDynamicLoRA(RobertaForSequenceClassification):
    def __init__(self, config, lora_r=1, hypernet_hidden_dim=16):
        super().__init__(config)
        self.config.output_hidden_states = True 
        self.hidden_size = config.hidden_size
        self.lora_r = lora_r

        # Your hypernetwork that outputs (A, B)
        self.hypernet = LoRAHyperNet(self.hidden_size, hypernet_hidden_dim, lora_r)

    # def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):

    #     output = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    #     for name, param in self.roberta.named_parameters():
    #         if 'hypernet' in name and param.requires_grad:
    #             print(name, torch.mean(param.grad) if param.grad is not None else "Ni mo")

    #     # print(self.hypernet.fc1.weight[0][0])

    #     return output

# %%
base_model = RobertaForSequenceClassification.from_pretrained(model_name)

lora_r = 1
base_hidden_size = 768
hypernet_hidden_dim = 16

hypernet = LoRAHyperNet(base_hidden_size, hypernet_hidden_dim, lora_r)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=lora_r,
    lora_alpha=16
)

def forward_hook(module, input, output):
    if hasattr(module, 'weight'):
        print(f"[WEIGHTS] {module.__class__.__name__} weight mean: {module.weight.data[0][:10]}")
    
def grad_hook(grad):
    print(f"[BACKWARD] grad shape: {grad.shape}, mean: {grad.mean().item():.8f}")
    return grad
    
def grad_hook_h(grad):
    print(f"[BACKWARD] grad shape: {grad.shape}, mean: {grad.mean().item():.8f} HYPERNET")
    return grad

base_model = get_peft_model(base_model, peft_config=peft_config)

target_layer = hypernet.fc1  # Example layer
# hook_handle = target_layer.register_forward_hook(forward_hook)
# target_layer.weight.register_hook(grad_hook_h)
# base_model.roberta.encoder.layer[-2].attention.self.query.lora_A["default"].weight.register_hook(grad_hook)
# base_model.roberta.encoder.layer[-3].attention.self.query.lora_A["default"].weight.register_hook(grad_hook)

dynamic_lora_layer = DynamicLoRALayer(base_hidden_size, lora_r, hypernet)

adapter_name = "default"

base_model.roberta.encoder.layer[-1].attention.self.query.lora_A[adapter_name] = dynamic_lora_layer
base_model.roberta.encoder.layer[-1].attention.self.query.lora_B[adapter_name] = nn.Identity()

# for param in base_model.hypernet.parameters():
#     param.requires_grad = True

# base_model.roberta.encoder.layer[-1].attention.self.query.lora_A.default.weight.requires_grad = False
# base_model.roberta.encoder.layer[-1].attention.self.query.lora_B.default.weight.requires_grad = False

total_params = sum(p.numel() for p in base_model.parameters())
trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

print(base_model.roberta.encoder.layer[-2])

# %%
for name, param in base_model.named_parameters():
    if "hypernet" in name:
        print(name, param.requires_grad)

# %%
training_args = TrainingArguments(
    output_dir="./outputs/roberta_base_cola_hy",
    eval_strategy="epoch",
    # eval_steps=10,
    save_strategy="steps",
    save_steps=1000000000,
    learning_rate=4e-4,
    per_device_train_batch_size=16, # 16
    gradient_accumulation_steps=2, # 2
    per_device_eval_batch_size=32,
    num_train_epochs=80, # 80
    logging_dir="./logs/roberta_base_cola_hy",
    metric_for_best_model="matthews_correlation",
    dataloader_num_workers=4,
    warmup_ratio=0.06,
    lr_scheduler_type="linear",
    optim="adamw_torch",
    weight_decay=0.1,
    disable_tqdm=True,
    # remove_unused_columns=False
)

# %%
trainer = Trainer(
    model=base_model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# %%
trainer.train()


