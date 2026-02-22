import argparse
import math
import os
from pathlib import Path
from typing import TypedDict

import torch
from bracex import expand
from const import layer_to_index
from loader import load_activations
from tqdm.auto import tqdm
from utils import TqdmLoggingHandler, get_project_dir, set_deterministic


class Args(TypedDict):
    output_type: str
    hidden_dim: int
    layers: list[str]
    dataset_configs: list[str]
    in_dir: Path
    in_path: Path
    out_dir: Path
    out_path: Path


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Count sae features from a dataset for particular layers and languages."
    )

    parser.add_argument(
        "--output-type",
        help="Output type of the loaded file",
        type=str,
        choices=[
            "EncoderOutput",
        ],
    )

    parser.add_argument(
        "--hidden-dim",
        help="hidden dimension of the model",
        type=int,
    )

    parser.add_argument(
        "--layer",
        help="layer(s) to be processed. The values should be the path to the layer in the model. Support bracex expansion",
        type=str,
        default=[],
        nargs="+",
    )

    parser.add_argument(
        "--dataset-configs",
        help='dataset configurations "dataset_name:config_name". Support bracex expansion',
        type=str,
        default=[],
        nargs="+",
    )

    parser.add_argument(
        "--in-dir",
        help="input directory",
        type=Path,
        default=get_project_dir(),
    )

    parser.add_argument(
        "--in-path",
        help="input path",
        type=Path,
    )

    parser.add_argument(
        "--out-dir",
        help="output directory",
        type=Path,
        default=get_project_dir(),
    )

    parser.add_argument(
        "--out-path",
        help="output path",
        type=Path,
    )

    args = parser.parse_args()

    processed_layers = []

    for layer in args.layer:
        processed_layers.extend(expand(layer))

    processed_dataset_configs = []

    for dataset_config in args.dataset_configs:
        processed_dataset_configs.extend(expand(dataset_config))

    return {
        "output_type": args.output_type,
        "hidden_dim": args.hidden_dim,
        "dataset_configs": processed_dataset_configs,
        "layers": processed_layers,
        "in_dir": args.in_dir,
        "in_path": args.in_path,
        "out_dir": args.out_dir,
        "out_path": args.out_path,
    }


def count_sae_features(
    activations_list: list,
    layer_index: int,
    over_zero_token: torch.Tensor,
    over_zero_example: torch.Tensor,
    over_zero_total: torch.Tensor,
    num_examples: int,
    num_tokens: int,
    max_active_over_zero: torch.Tensor,
    min_active_over_zero: torch.Tensor,
    output_type: str,
    rounding_digit: int = 3,
):
    if output_type == "EncoderOutput":
        for activations in activations_list:
            # Convert sparse tensors to dense for processing for JumpRelu activations
            top_acts = activations.top_acts.to_dense()
            top_indices = activations.top_indices

            num_examples += 1
            num_tokens += top_indices.shape[1]  # (batch, tokens, top_K)

            flat_top_indices = top_indices.flatten()
            flat_top_acts = top_acts.flatten()
            
            # TopK -> (batch, tokens, top_K)
            # JumpRelu -> (batch, tokens, max_K). Need to remove zero activations
            non_zero_flat_top_acts = flat_top_acts > 0
            flat_top_indices = flat_top_indices[non_zero_flat_top_acts]
            flat_top_acts = flat_top_acts[non_zero_flat_top_acts]

            unique_feature_indices = torch.unique(flat_top_indices)
            over_zero_example[layer_index, unique_feature_indices] += 1

            ones = torch.ones_like(flat_top_indices, dtype=over_zero_token.dtype)
            over_zero_token[layer_index].scatter_add_(0, flat_top_indices, ones)

            over_zero_total[layer_index].scatter_add_(0, flat_top_indices, flat_top_acts)

            rounded_acts = flat_top_acts.round(decimals=rounding_digit)

            max_active_over_zero[layer_index].scatter_reduce_(
                0,
                flat_top_indices,
                rounded_acts,
                reduce="amax",
                include_self=True,
            )

            min_active_over_zero[layer_index].scatter_reduce_(
                0,
                flat_top_indices,
                rounded_acts,
                reduce="amin",
                include_self=True,
            )

    return (
        over_zero_token,
        over_zero_example,
        over_zero_total,
        num_examples,
        num_tokens,
        max_active_over_zero,
        min_active_over_zero,
    )


def main(args: Args):
    num_layers = len(args["layers"])
    hidden_dim = args["hidden_dim"]

    for dataset_config in tqdm(args["dataset_configs"], desc="Processing languages"):
        dataset, lang = dataset_config.split(":")

        over_zero_token = torch.zeros((num_layers, hidden_dim))
        over_zero_example = torch.zeros((num_layers, hidden_dim))
        over_zero_total = torch.zeros((num_layers, hidden_dim))
        max_active_over_zero = torch.zeros((num_layers, hidden_dim))
        min_active_over_zero = torch.full((num_layers, hidden_dim), math.inf)

        for layer in tqdm(args["layers"], desc="Processing layers", leave=False):
            num_examples = 0
            num_tokens = 0

            input_dir = args["in_dir"] / args["in_path"] / dataset / lang
            activations = load_activations(input_dir, layer, logger)

            layer_index = layer_to_index[layer]
            (
                over_zero_token,
                over_zero_example,
                over_zero_total,
                num_examples,
                num_tokens,
                max_active_over_zero,
                min_active_over_zero,
            ) = count_sae_features(
                activations,
                layer_index,
                over_zero_token,
                over_zero_example,
                over_zero_total,
                num_examples,
                num_tokens,
                max_active_over_zero,
                min_active_over_zero,
                args["output_type"],
            )

        output_dir = args["out_dir"] / args["out_path"] / dataset
        file_path = output_dir / f"{lang}.pt"

        os.makedirs(output_dir, exist_ok=True)

        # Save as a sparse tensor
        min_active_over_zero[min_active_over_zero == math.inf] = 0

        output = {
            "num_examples": num_examples,
            "num_tokens": num_tokens,
            "over_zero_token": over_zero_token.to_sparse(),
            "over_zero_example": over_zero_example.to_sparse(),
            "over_zero_total": over_zero_total.to_sparse(),
            "max_active_over_zero": max_active_over_zero.to_sparse(),
            "min_active_over_zero": min_active_over_zero.to_sparse(),
        }

        torch.save(output, file_path)


if __name__ == "__main__":
    set_deterministic()

    logger = TqdmLoggingHandler.get_logger("activations_count")

    args = parse_args()
    main(args)
