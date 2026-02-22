from pathlib import Path
from typing import TypedDict

import torch
from datasets import Dataset
from loader import load_sae
from nnsight import LanguageModel
from sparsify.sparse_coder import EncoderOutput
from tqdm.auto import tqdm
from utils import get_device, get_nested_attr


def collect_activations(
    llm: LanguageModel, layers: list[str], prompt: str
) -> dict[str, list[torch.Tensor]]:
    layers_modules = {layer: get_nested_attr(llm, layer) for layer in layers}

    layers_activations = {layer: None for layer in layers}

    with llm.trace(prompt):
        for layer in layers:
            layers_activations[layer] = layers_modules[layer].output.cpu().save()

    layers_activations_processed = {
        layer: (
            activations[0].value
            if isinstance(activations, tuple)
            else activations.value
        )
        for layer, activations in layers_activations.items()
    }

    return layers_activations_processed


def sae_features_from_activations(
    activations_list: list[torch.Tensor],
    sae: any,
    sae_model_name: str,
    device: torch.device,
    batch: int = 100,
):
    activations_size = [
        activations.shape[1] for activations in activations_list
    ]  # [a, b, ...]

    if sae_model_name.startswith("EleutherAI/"):
        activations_list = torch.cat(
            activations_list, dim=1
        )  # tensor(1, a+b+..., 2048)
        top_acts = []
        top_indices = []
        # list[tensor(1, batch, 2048), ...]
        chunks = torch.split(activations_list, batch, dim=1)

        for chunk in chunks:
            sae_features = sae.encode(chunk.squeeze(0).to(device))
            top_acts.append(sae_features.top_acts.unsqueeze(0).cpu())
            top_indices.append(sae_features.top_indices.unsqueeze(0).cpu())

        top_acts = torch.cat(top_acts, dim=1)
        top_acts = torch.split(top_acts, activations_size, dim=1)

        top_indices = torch.cat(top_indices, dim=1)
        top_indices = torch.split(top_indices, activations_size, dim=1)

        all_sae_features = []

        for top_acts, top_indices in zip(top_acts, top_indices):
            all_sae_features.append(EncoderOutput(top_acts, top_indices, None))

        return all_sae_features
    elif sae_model_name.startswith("llama_scope") or sae_model_name.startswith(
        "gemma-scope"
    ):
        all_sae_features = []

        for activations in activations_list:
            sae_feature_activations_tensor = sae.encode(activations.to(device=device))
            
            # Get the maximum number of non-zero activations across all samples and tokens
            max_k = (sae_feature_activations_tensor > 0).sum(dim=-1).max().item()
            
            # Keep only the top max_k activations and their indices for each token
            top_acts, top_indices = torch.topk(sae_feature_activations_tensor, k=max_k, dim=-1)
            top_acts = top_acts.cpu()
            top_indices = top_indices.cpu()

            # Save as sparse tensors to save memory
            all_sae_features.append(
                EncoderOutput(top_acts.to_sparse(), top_indices, None)
            )

        return all_sae_features


class DatasetConfig(TypedDict):
    name: str
    lang: str
    split: str
    start: int
    end: int | float


def collect_all_activations(
    llm: LanguageModel,
    layers: list[str],
    dataset: Dataset,
    prompt_template: str,
) -> dict[str, list[torch.Tensor]]:
    all_activations = {layer: [] for layer in layers}

    for row in tqdm(dataset, desc="Processing Samples", leave=False):
        prompt = prompt_template.format_map(row)
        activations = collect_activations(llm, layers, prompt)

        for layer, layer_activations in activations.items():
            all_activations[layer].append(layer_activations)

    return all_activations


def collect_all_sae_features(
    all_activations: dict[str, list[torch.Tensor]],
    layers: list[str],
    model: str,
    sae_model: str,
    local_sae_dir: Path,
    batch: int,
):
    all_sae_features = {layer: [] for layer in layers}

    device = get_device()

    for layer, layer_activations in all_activations.items():
        sae = load_sae(model, sae_model, layer, local_sae_dir).to(device)
        all_sae_features[layer] = sae_features_from_activations(
            layer_activations, sae, sae_model, device, batch
        )

    return all_sae_features
