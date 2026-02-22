#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"

python "$DIR/../../scripts/classify.py" MartinThoma/wili_2018 \
    --model google/gemma-2-2b \
    --split test \
    --lang eng deu fra ita por hin spa tha bul rus tur vie jpn kor zho \
    --layer model.layers.{0..25}.post_feedforward_layernorm \
    --start 0 --end 500 \
    --sae-model gemma-scope-2b-pt-mlp-canonical \
    --batch 500 \
    --lape-result-path 'sae_features_specific/google/gemma-2-2b/gemma-scope-2b-pt-mlp-canonical/lape_all.pt' \
    --classifier-type sae-count

python "$DIR/../../scripts/classify.py" MartinThoma/wili_2018 \
    --model google/gemma-2-2b \
    --split test \
    --lang eng deu fra ita por hin spa tha bul rus tur vie jpn kor zho \
    --layer model.layers.{0..25}.mlp.act_fn \
    --start 0 --end 500 \
    --lape-result-path 'mlp_acts_specific/google/gemma-2-2b/lape_neuron.pt' \
    --classifier-type neuron-count

python "$DIR/../../scripts/classify.py" MartinThoma/wili_2018 \
    --split test \
    --lang eng deu fra ita por hin spa tha bul rus tur vie jpn kor zho \
    --start 0 --end 500 \
    --classifier-type fasttext
