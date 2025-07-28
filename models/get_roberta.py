import torch
import torch.nn as nn
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
from transformers import (RobertaForSequenceClassification, RobertaModel,
                          RobertaTokenizer, Trainer, TrainingArguments)

from models.dynamic_lora_layer import DynamicLoRALayer
from models.hypernet import LoRAHyperNet


def get_baseline_roberta(model_name="roberta-base", lora_r=1, lora_alpha=16):
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha
    )

    model = get_peft_model(model, peft_config=peft_config)

    return model, tokenizer

def get_hypernet_on_last_layer_roberta(model_name="roberta-base", lora_r=1, lora_alpha=16, hypernet_hidden_dim=16, hypernet_embeddings_dim=8, use_on_value_matrix=True):
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    base_hidden_size = 768 #! This must be set according to the layer that we are applying hypernet on

    hypernet = LoRAHyperNet(base_hidden_size, hypernet_hidden_dim, lora_r, num_of_embeddings=2, embedding_dim=hypernet_embeddings_dim)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha
    )

    model = get_peft_model(model, peft_config=peft_config)

    dynamic_lora_layer_0 = DynamicLoRALayer(base_hidden_size, lora_r, hypernet, layer_id=0)
    dynamic_lora_layer_1 = DynamicLoRALayer(base_hidden_size, lora_r, hypernet, layer_id=1)

    adapter_name = "default"

    model.roberta.encoder.layer[-1].attention.self.query.lora_A[adapter_name] = dynamic_lora_layer_0
    model.roberta.encoder.layer[-1].attention.self.query.lora_B[adapter_name] = nn.Identity()

    if use_on_value_matrix:
        model.roberta.encoder.layer[-1].attention.self.value.lora_A[adapter_name] = dynamic_lora_layer_1
        model.roberta.encoder.layer[-1].attention.self.value.lora_B[adapter_name] = nn.Identity()

    return model, tokenizer, hypernet