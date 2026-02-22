#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"

python "$DIR/../../scripts/sae_features_count.py" \
    --output-type "EncoderOutput" \
    --hidden-dim 131072 \
    --dataset-configs 'facebook/xnli:{en,de,fr,hi,es,th,bg,ru,tr,vi}' 'google-research-datasets/paws-x:{en,de,fr,es,ja,ko,zh}' 'openlanguagedata/flores_plus:{eng_Latn,deu_Latn,fra_Latn,ita_Latn,por_Latn,hin_Deva,spa_Latn,tha_Thai,bul_Cyrl,rus_Cyrl,tur_Latn,vie_Latn,jpn_Jpan,kor_Hang,cmn_Hans}' \
    --layer 'model.layers.{0..41}.post_feedforward_layernorm' \
    --in-path 'sae_features/google/gemma-2-9b/gemma-scope-9b-pt-mlp-canonical' \
    --out-path 'sae_features_count/google/gemma-2-9b/gemma-scope-9b-pt-mlp-canonical'
