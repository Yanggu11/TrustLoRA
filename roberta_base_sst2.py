from datasets import load_dataset
import numpy as np
import evaluate
from peft import get_peft_model, LoraConfig, TaskType
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

for it in range(3):
    print(f"====== Run {it} ===============")

    # Load tokenizer and base model
    model_name = "roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    base_model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Apply PEFT with LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=8
    )
    model = get_peft_model(base_model, peft_config)

    # Print parameter stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Load SST-2 dataset
    dataset = load_dataset("glue", "sst2")

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=512)

    encoded_dataset = dataset.map(tokenize_function, batched=True)
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Load SST-2 metric (accuracy)
    metric = evaluate.load("glue", "sst2")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./outputs/roberta_base_sst2",
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=1000000,  # Effectively disables checkpoint saving
        learning_rate=5e-4,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=32,
        num_train_epochs=60,
        logging_dir=f"./logs/roberta_base_sst2",
        metric_for_best_model="accuracy",
        greater_is_better=True,
        warmup_ratio=0.06,
        lr_scheduler_type="linear",
        optim="adamw_torch",
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

    # Train
    trainer.train()
