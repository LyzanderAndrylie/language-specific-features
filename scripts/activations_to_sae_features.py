import argparse
from math import inf
from pathlib import Path
from typing import TypedDict

import torch
from bracex import expand
from collector import collect_all_activations, collect_all_sae_features
from const import (
    dataset_choices,
    lang_choices,
    model_choices,
    prompt_templates,
    sae_model_choices,
)
from loader import load_dataset_specific
from nnsight import LanguageModel
from utils import TqdmLoggingHandler, get_project_dir, set_deterministic
from writer import save_activations


class DatasetArgs(TypedDict):
    name: str
    start: int
    end: int | float
    split: str


class Args(TypedDict):
    model: str
    dataset: DatasetArgs
    languages: list[str]
    layers: list[str]
    sae_model: str
    local_sae_dir: Path | None
    batch: int
    out_dir: Path


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Collect activations from a dataset for particular layers and languages and transform it into SAE features."
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
        "--split",
        help="dataset split",
        type=str,
        default="train",
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
        "--start",
        help="sample start index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--end",
        help="sample end index",
        type=int,
        default=inf,
    )

    parser.add_argument(
        "--sae-model",
        help="sae model name",
        type=str,
        default=None,
        choices=sae_model_choices,
    )

    parser.add_argument(
        "--local-sae-dir",
        help="local directory to load SAE model from",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--batch",
        help="batch size",
        type=int,
        default=100,
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
        "dataset": {
            "name": args.dataset,
            "start": args.start,
            "end": args.end,
            "split": args.split,
        },
        "languages": args.lang,
        "layers": processed_layers,
        "sae_model": args.sae_model,
        "local_sae_dir": args.local_sae_dir,
        "batch": args.batch,
        "out_dir": args.out_dir,
    }


@torch.inference_mode()
def main(args: Args):
    logger.info(f"Loading Model: {args['model']}")

    llm = LanguageModel(args["model"], device_map="auto", dispatch=True)

    for lang in args["languages"]:
        dataset_config = {
            **args["dataset"],
            "lang": lang,
        }

        logger.info(
            f"Loading Dataset: {dataset_config['name']} ({dataset_config['lang']})"
        )

        dataset = load_dataset_specific(
            dataset_config["name"],
            dataset_config["lang"],
            dataset_config["split"],
            dataset_config["start"],
            dataset_config["end"],
        )

        prompt_template = prompt_templates[dataset_config["name"]][
            dataset_config["lang"]
        ]

        all_activations = collect_all_activations(
            llm,
            args["layers"],
            dataset,
            prompt_template,
        )

        all_sae_features = collect_all_sae_features(
            all_activations,
            args["layers"],
            args["model"],
            args["sae_model"],
            args["local_sae_dir"],
            args["batch"],
        )

        # Save sae features to disk
        output_dir = (
            args["out_dir"]
            / "sae_features"
            / args["model"]
            / args["sae_model"]
            / args["dataset"]["name"]
            / lang
        )
        save_activations(
            output_dir,
            all_sae_features,
            args["dataset"]["start"],
            args["dataset"]["end"],
        )

        del all_activations
        del all_sae_features


if __name__ == "__main__":
    logger = TqdmLoggingHandler.get_logger("activations_to_sae_features")

    set_deterministic()

    args = parse_args()
    main(args)
