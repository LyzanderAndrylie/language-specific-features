#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"

python "$DIR/../../scripts/classify.py" MartinThoma/wili_2018 \
    --model meta-llama/Llama-3.1-8B \
    --split test \
    --lang eng deu fra ita por hin spa tha bul rus tur vie jpn kor zho \
    --layer model.layers.{0..31}.mlp \
    --start 0 --end 500 \
    --sae-model llama_scope_lxm_32x \
    --batch 500 \
    --lape-result-path 'sae_features_specific/meta-llama/Llama-3.1-8B/llama_scope_lxm_32x/lape_all.pt' \
    --classifier-type sae-count

python "$DIR/../../scripts/classify.py" MartinThoma/wili_2018 \
    --model meta-llama/Llama-3.1-8B \
    --split test \
    --lang eng deu fra ita por hin spa tha bul rus tur vie jpn kor zho \
    --layer model.layers.{0..31}.mlp.act_fn \
    --start 0 --end 500 \
    --lape-result-path 'mlp_acts_specific/meta-llama/Llama-3.1-8B/lape_neuron.pt' \
    --classifier-type neuron-count

python "$DIR/../../scripts/classify.py" MartinThoma/wili_2018 \
    --split test \
    --lang eng deu fra ita por hin spa tha bul rus tur vie jpn kor zho \
    --start 0 --end 500 \
    --classifier-type fasttext
