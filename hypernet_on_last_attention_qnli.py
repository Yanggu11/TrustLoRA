# %%
from datasets import load_dataset
import numpy as np
import evaluate
import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig

print("HYPERNET Roberta_base_QNLI")
for i in range(3):
    print(f"====== Run {i} ===============")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "roberta-base"

    # %%
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=512)

    dataset = load_dataset("glue", "qnli")

    encoded_dataset = dataset.map(tokenize_function, batched=True)
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    metric = evaluate.load("glue", "qnli")

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
    class RobertaWithDynamicLoRA(RobertaForSequenceClassification):
        def __init__(self, config, lora_r=1, hypernet_hidden_dim=16):
            super().__init__(config)
            self.config.output_hidden_states = True 
            self.hidden_size = config.hidden_size
            self.lora_r = lora_r

            # Create hypernetwork
            self.hypernet = LoRAHyperNet(self.hidden_size, hypernet_hidden_dim, lora_r)

            # Use fixed A in eval, learned through computation in training
            self.register_buffer("A_fixed", torch.randn(lora_r, self.hidden_size))

            self.dropout = nn.Dropout(config.hidden_dropout_prob)

        def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
            device = input_ids.device
            
            # print(self.hypernet.fc1.weight[:5][:5]) # check if the gradients flow properly throught the hypernetwork
            
            init_A_B = torch.randn((2, self.lora_r, self.hidden_size), device=device) # [r, hidden_size * 2]
            A, B = self.hypernet(init_A_B)

            print(A.shape, B.shape)

            # Forward pass through model to get hidden states
            outputs = self.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]

            # Apply dynamic LoRA on final layer hidden states
            # hidden + (hidden @ A.T @ B.T)
            lora_out = torch.matmul(hidden_states, A.T)     # [batch, seq_len, r]
            lora_out = torch.matmul(lora_out, B.T)          # [batch, seq_len, hidden_size]
            adapted_hidden = hidden_states + lora_out

            # Use [CLS] token from adapted hidden
            cls_output = adapted_hidden[:, 0]  # [batch, hidden]
            logits = self.dropout(cls_output)
            logits = self.classifier.out_proj(logits)

            loss = None
            if labels is not None:
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)

            return {"logits": logits, "loss": loss} if loss is not None else {"logits": logits}


    # %%
    base_model = RobertaWithDynamicLoRA.from_pretrained(model_name)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=base_model.lora_r,
        lora_alpha=16
    )

    base_model = get_peft_model(base_model, peft_config=peft_config)

    for param in base_model.hypernet.parameters():
        param.requires_grad = True

    base_model.roberta.encoder.layer[-1].attention.self.query.lora_A.default.weight.requires_grad = False
    base_model.roberta.encoder.layer[-1].attention.self.query.lora_B.default.weight.requires_grad = False

    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # %%
    for name, param in base_model.named_parameters():
        if "hypernet" in name:
            print(name, param.requires_grad)

    # %%
    print(base_model.roberta.encoder.layer[-1].attention.self.query.lora_B.default.weight.shape)

    # %%
    training_args = TrainingArguments(
            output_dir="./outputs/roberta_base_qnli",
            eval_strategy="epoch",
            save_strategy="steps",
            save_steps=1000000,
            learning_rate=4e-4,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=2,
            per_device_eval_batch_size=32,
            num_train_epochs=25,
            logging_dir="./logs/roberta_base_qnli",
            metric_for_best_model="accuracy",  # QNLI uses accuracy metric primarily
            dataloader_num_workers=4,
            warmup_ratio=0.06,
            weight_decay=0.1,
            lr_scheduler_type="linear",
            optim="adamw_torch",
            disable_tqdm=True
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


