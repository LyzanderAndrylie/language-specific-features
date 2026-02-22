import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from bracex import expand
from const import dataset_choices, lang_choices, model_choices, sae_model_choices
from loader import load_activations
from tqdm.auto import tqdm
from utils import TqdmLoggingHandler, get_project_dir, set_deterministic


class Args(TypedDict):
    model: str
    dataset: str
    languages: list[str]
    layers: list[str]
    sae_model: str
    in_dir: Path
    out_dir: Path


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Visualize sae features from a dataset for particular layers and languages."
    )

    parser.add_argument(
        "model",
        help="model name",
        type=str,
        choices=model_choices,
    )

    parser.add_argument(
        "dataset",
        help="dataset name",
        type=str,
        choices=dataset_choices,
    )

    parser.add_argument(
        "--lang",
        help="language(s) to be processed",
        type=str,
        default=[],
        nargs="+",
        choices=lang_choices,
    )

    parser.add_argument(
        "--layer",
        help="layer(s) to be processed. The values should be the path to the layer in the model. Support bracex expansion",
        type=str,
        default=[],
        nargs="+",
    )

    parser.add_argument(
        "--sae-model",
        help="sae model name",
        type=str,
        default=None,
        choices=sae_model_choices,
    )

    parser.add_argument(
        "--in-dir",
        help="input directory",
        type=Path,
        default=get_project_dir(),
    )

    parser.add_argument(
        "--out-dir",
        help="output directory",
        type=Path,
        default=get_project_dir(),
    )

    args = parser.parse_args()

    processed_layers = []

    for layer in args.layer:
        processed_layers.extend(expand(layer))

    return {
        "model": args.model,
        "dataset": args.dataset,
        "languages": args.lang,
        "layers": processed_layers,
        "sae_model": args.sae_model,
        "in_dir": args.in_dir,
        "out_dir": args.out_dir,
    }


def extract_features(sae_model: str, sae_features: any):
    if sae_model.startswith("EleutherAI/"):
        return sae_features.top_acts, sae_features.top_indices
    elif sae_model.startswith("llama_scope") or sae_model.startswith(
        "gemma-scope"
    ):
        pass


def process_sae_features(
    sae_features_list: list[any],
    sae_model: str,
    layer: str,
    lang: str,
    rounding_digit=3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sae_feature_index_to_activations = defaultdict(list)
    sae_feature_index_to_dataset_id_token_id_act_val = defaultdict(list)

    for dataset_row_index, sae_features in enumerate(sae_features_list):
        top_acts, top_indices = extract_features(sae_model, sae_features)
        top_act_index_per_token = zip(top_acts.squeeze(0), top_indices.squeeze(0))

        for token_index, (top_act, top_index) in enumerate(top_act_index_per_token):
            for act_val, feature_index in zip(top_act.tolist(), top_index.tolist()):
                sae_feature_index_to_activations[feature_index].append(act_val)
                dataset_id_token_id_act_val = (
                    dataset_row_index,
                    token_index,
                    round(act_val, rounding_digit),
                )
                sae_feature_index_to_dataset_id_token_id_act_val[feature_index].append(
                    dataset_id_token_id_act_val
                )

    sae_features_count = {}
    sae_features_avg = {}
    sae_features_q1 = {}
    sae_features_median = {}
    sae_features_q3 = {}
    sae_features_min_active = {}
    sae_features_max_active = {}
    sae_features_std = {}

    for feature_index, activations in sae_feature_index_to_activations.items():
        sae_features_count[feature_index] = len(activations)
        sae_features_avg[feature_index] = round(
            np.mean(activations).item(), rounding_digit
        )
        sae_features_q1[feature_index] = round(
            np.percentile(activations, 25).item(), rounding_digit
        )
        sae_features_median[feature_index] = round(
            np.median(activations).item(), rounding_digit
        )
        sae_features_q3[feature_index] = round(
            np.percentile(activations, 75).item(), rounding_digit
        )
        sae_features_min_active[feature_index] = round(
            np.min(activations).item(), rounding_digit
        )
        sae_features_max_active[feature_index] = round(
            np.max(activations).item(), rounding_digit
        )
        sae_features_std[feature_index] = round(
            np.std(activations).item(), rounding_digit
        )

    # Create a dataframe from the statistics
    statistics = {
        "count": sae_features_count,
        "avg": sae_features_avg,
        "q1": sae_features_q1,
        "median": sae_features_median,
        "q3": sae_features_q3,
        "min_active": sae_features_min_active,
        "max_active": sae_features_max_active,
        "std": sae_features_std,
        "lang": lang,
        "layer": layer,
    }

    df_statistics = pd.DataFrame(statistics)
    df_statistics.sort_index(inplace=True)
    df_statistics.reset_index(inplace=True)

    # Create a dataframe from the dataset_token_activations
    dataset_token_activations = {
        "count": sae_features_count,
        "dataset_row_id_token_id_act_val": sae_feature_index_to_dataset_id_token_id_act_val,
    }

    df_dataset_token_activations = pd.DataFrame(dataset_token_activations)
    df_dataset_token_activations.sort_index(inplace=True)
    df_dataset_token_activations.reset_index(inplace=True)

    return df_statistics, df_dataset_token_activations


def main(args: Args):
    for lang in tqdm(args["languages"], desc="Processing languages"):
        for layer in tqdm(args["layers"], desc="Processing layers", leave=False):
            input_dir = (
                args["in_dir"]
                / "sae_features"
                / args["model"]
                / args["sae_model"]
                / args["dataset"]
                / lang
            )

            sae_features = load_activations(input_dir, layer, logger)
            df_statistics, df_dataset_token_activations = process_sae_features(
                sae_features, args["sae_model"], layer, lang
            )

            # Save the statistics
            output_dir = (
                args["out_dir"]
                / "statistics"
                / args["model"]
                / args["sae_model"]
                / args["dataset"]
                / "summary"
                / layer
            )
            os.makedirs(output_dir, exist_ok=True)

            df_statistics.to_csv(output_dir / f"{lang}.csv", index=False)

            # Save the dataset_token_activations
            output_dir = (
                args["out_dir"]
                / "statistics"
                / args["model"]
                / args["sae_model"]
                / args["dataset"]
                / "dataset_token_activations"
                / layer
            )
            os.makedirs(output_dir, exist_ok=True)

            df_dataset_token_activations.to_csv(output_dir / f"{lang}.csv", index=False)


if __name__ == "__main__":
    set_deterministic()

    logger = TqdmLoggingHandler.get_logger("statistics")

    args = parse_args()
    main(args)
