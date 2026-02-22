#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"

python "$DIR/../../scripts/identify.py" \
    --model 'google/gemma-2-2b' \
    --dataset-configs 'facebook/xnli:{en,de,fr,hi,es,th,bg,ru,tr,vi}' 'google-research-datasets/paws-x:{en,de,fr,es,ja,ko,zh}' 'openlanguagedata/flores_plus:{eng_Latn,deu_Latn,fra_Latn,ita_Latn,por_Latn,hin_Deva,spa_Latn,tha_Thai,bul_Cyrl,rus_Cyrl,tur_Latn,vie_Latn,jpn_Jpan,kor_Hang,cmn_Hans}' \
    --in-path 'mlp_acts_count/google/gemma-2-2b' \
    --out-path 'mlp_acts_specific/google/gemma-2-2b' \
    --algorithm 'lape' \
    --out-filename 'lape_neuron.pt'
