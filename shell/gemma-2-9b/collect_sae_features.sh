#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"

python "$DIR/../../scripts/activations_to_sae_features.py" google/gemma-2-9b facebook/xnli \
    --split train \
    --lang en de fr hi es th bg ru tr vi \
    --layer model.layers.{0..41}.post_feedforward_layernorm \
    --start 0 --end 1000 \
    --sae-model gemma-scope-9b-pt-mlp-canonical \
    --batch 500

python "$DIR/../../scripts/activations_to_sae_features.py" google/gemma-2-9b google-research-datasets/paws-x \
    --split train \
    --lang en de fr es ja ko zh \
    --layer model.layers.{0..41}.post_feedforward_layernorm \
    --start 0 --end 1000 \
    --sae-model gemma-scope-9b-pt-mlp-canonical \
    --batch 500

python "$DIR/../../scripts/activations_to_sae_features.py" google/gemma-2-9b openlanguagedata/flores_plus \
    --split dev \
    --lang eng_Latn deu_Latn fra_Latn ita_Latn por_Latn hin_Deva spa_Latn tha_Thai bul_Cyrl rus_Cyrl tur_Latn vie_Latn jpn_Jpan kor_Hang cmn_Hans \
    --layer model.layers.{0..41}.post_feedforward_layernorm \
    --start 0 --end 997 \
    --sae-model gemma-scope-9b-pt-mlp-canonical \
    --batch 500
