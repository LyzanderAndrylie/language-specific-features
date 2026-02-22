#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"

python "$DIR/../../scripts/activations_to_sae_features.py" meta-llama/Llama-3.1-8B facebook/xnli \
    --split train \
    --lang en de fr hi es th bg ru tr vi \
    --layer model.layers.{0..31}.mlp \
    --start 0 --end 1000 \
    --sae-model llama_scope_lxm_32x \
    --batch 500

python "$DIR/../../scripts/activations_to_sae_features.py" meta-llama/Llama-3.1-8B google-research-datasets/paws-x \
    --split train \
    --lang en de fr es ja ko zh \
    --layer model.layers.{0..31}.mlp \
    --start 0 --end 1000 \
    --sae-model llama_scope_lxm_32x \
    --batch 500

python "$DIR/../../scripts/activations_to_sae_features.py" meta-llama/Llama-3.1-8B openlanguagedata/flores_plus \
    --split dev \
    --lang eng_Latn deu_Latn fra_Latn ita_Latn por_Latn hin_Deva spa_Latn tha_Thai bul_Cyrl rus_Cyrl tur_Latn vie_Latn jpn_Jpan kor_Hang cmn_Hans \
    --layer model.layers.{0..31}.mlp \
    --start 0 --end 997 \
    --sae-model llama_scope_lxm_32x \
    --batch 500
