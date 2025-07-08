from datasets import load_dataset
import numpy as np
import torch
import torch.nn as nn
import evaluate
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

from evaluation.metrices import compute_ece

print("Baseline CoLA")

for it in range(2):
    print(f"====== Run {it} ===============")
    #* Init finetuning using peft 

    # Load tokenizer and model
    model_name = "roberta-base" # "roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    base_model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=1,
        lora_alpha=16
    )
    model = get_peft_model(base_model, peft_config)

    # Total number of parameters
    total_params = sum(p.numel() for p in base_model.parameters())

    # Trainable parameters (typically only a subset with LoRA)
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Load CoLA dataset
    dataset = load_dataset("glue", "cola")

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=512)

    encoded_dataset = dataset.map(tokenize_function, batched=True)
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    metric = evaluate.load("glue", "cola")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

        predictions = np.argmax(probs, axis=-1)
        results = metric.compute(predictions=predictions, references=labels)

        # Compute ECE
        ece = compute_ece(probs, labels)
        results["ece"] = ece

        return results

    training_args = TrainingArguments(
        output_dir="./outputs/baseline_cola",
        eval_strategy="epoch",
        # eval_steps=10,
        save_strategy="steps",
        save_steps=100000000,
        learning_rate=4e-4,
        per_device_train_batch_size=16, #* maybe think about adding gradient accumulation
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=32,
        num_train_epochs=80,
        logging_dir="./logs/baseline_cola",
        logging_strategy="epoch",
        metric_for_best_model="matthews_correlation",
        dataloader_num_workers=4,
        warmup_ratio=0.06,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        weight_decay=0.1,
        disable_tqdm=True
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()