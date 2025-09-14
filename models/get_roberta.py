import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
)

from models.dynamic_lora_layer import DynamicLoRALayer
from models.hypernet import LoRAHyperNet


def get_baseline_roberta(
    model_name="roberta-base",
    use_peft=False,
    lora_r=1,
    lora_alpha=16,
    target_modules=["query", "value"],
    layers_to_transform=list(range(12)),
    layers_pattern="encoder.layer",
    layers_to_freeze=[],
):
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    if use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            layers_to_transform=layers_to_transform,
            layers_pattern=layers_pattern,
        )

        model = get_peft_model(model, peft_config=peft_config)

    for name, param in model.named_parameters():
        for layer in layers_to_freeze:
            if layer in str(name):
                param.requires_grad = False

    return model, tokenizer


def get_hypernet_on_last_layer_roberta(
    model_name="roberta-base",
    use_peft=True,
    lora_r=1,
    lora_alpha=16,
    hypernet_layers=[11],
    hypernet_hidden_dim=16,
    hypernet_embeddings_dim=8,
    use_on_value_matrix=True,
    hypernet_with_embedding_input_only=False, #! in older versions of the code there is wihtout "_only"
    use_fixed_A=True,
    use_large_model=False,
    target_modules=["query", "value"],
    layers_to_transform=list(range(12)),
    layers_pattern="encoder.layer",
    layers_to_freeze=[],
):
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    base_hidden_size = model.config.hidden_size

    hypernet = LoRAHyperNet(
        base_hidden_size,
        hypernet_hidden_dim,
        lora_r,
        num_of_embeddings=2 * len(hypernet_layers) if use_on_value_matrix else len(hypernet_layers),
        embedding_dim=hypernet_embeddings_dim,
        embedding_input_only=hypernet_with_embedding_input_only,
        large_model=use_large_model
    )
    if use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            layers_to_transform=layers_to_transform,
            layers_pattern=layers_pattern,
        )

        model = get_peft_model(model, peft_config=peft_config)

    dynamic_lora_layers = []

    for idx, layer_id in enumerate(hypernet_layers):

        dynamic_lora_layers.append(DynamicLoRALayer(
            base_hidden_size, lora_r, hypernet, layer_id=idx, use_fixed_A=use_fixed_A
        ))

        adapter_name = "default"

        model.roberta.encoder.layer[layer_id].attention.self.query.lora_A[
            adapter_name
        ] = dynamic_lora_layers[-1]
        model.roberta.encoder.layer[layer_id].attention.self.query.lora_B[
            adapter_name
        ] = nn.Identity()

        if use_on_value_matrix:
            dynamic_lora_layers.append(DynamicLoRALayer(
                base_hidden_size, lora_r, hypernet, layer_id=idx + len(hypernet_layers), use_fixed_A=use_fixed_A
            ))
            model.roberta.encoder.layer[layer_id].attention.self.value.lora_A[
                adapter_name
            ] = dynamic_lora_layers[-1]
            model.roberta.encoder.layer[layer_id].attention.self.value.lora_B[
                adapter_name
            ] = nn.Identity()

    for name, param in model.named_parameters():
        for layer in layers_to_freeze:
            if layer in str(name):
                param.requires_grad = False

    return model, tokenizer, hypernet
