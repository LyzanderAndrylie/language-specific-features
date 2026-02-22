# Language-Specific Features

This repository is the official implementation of our paper [Sparse Autoencoders Can Capture Language-Specific Concepts Across Diverse Languages](https://arxiv.org/abs/2507.11230).

![Language-Specific Features](./images/Alternative%20Illustration.svg)

## Language-Specific Features by SAE-LAPE

We provide the identified language-specific features for Llama 3.2 1B and Gemma 2 in the `compressed` directory. The features are identified using the SAE-LAPE method, which is described in the paper.

In addition, we provide the identfied langauge-specific neurons using the [LAPE](https://aclanthology.org/2024.acl-long.309/) method for Llama 3.2 1B and Gemma 2 in the `compressed` directory for comparison. We use the code from [Language-Specific-Neurons](https://github.com/RUCAIBox/Language-Specific-Neurons) for the LAPE method and modify it to work with SAE features, which we refer to as SAE-LAPE.

## Visualization

All visualizations can be accessed interactively at <https://lyzanderandrylie.github.io/language-specific-features/> and are also available in the `images` directory.

## Scripts

We provide several scripts to collect, process, and analyze language-specific features in the `scripts` directory. We also provide shell scripts to run the scripts in the `shell` directory.

### Sparse Autoencoder (SAE)

1. Collect SAE Features from Activations of an LLM

   ```bash
   python activations_to_sae_features.py <model_name> <dataset_name> \
        --split <dataset_split_name> \
        --lang <lang_1> ... \
        --layer <layer_1> ... \
        --start <row_number> --end <row_number> \
        --sae-model <sae_model_name> \
        --local-sae-dir <local_directory> \
        --batch <batch_size> \
        --out-dir <output_directory>
   ```

   > Note:
   > `sae-model` will be load from the Hugging Face model hub if `--local-sae-dir` option is not provided.
   > Support [bracex](https://github.com/facelessuser/bracex) for layer names.

2. Processing SAE Features

   ```bash
   python sae_statistics.py <model_name> <dataset_name> \
       --lang <lang_1> ... \
       --layer <layer_1> ... \
       --sae-model <sae_model_name> \
       --in-dir <input_directory> \
       --out-dir <output_directory>
   ```

3. Count SAE Features

   ```bash
   python sae_features_count.py \
      --output-type <output_type> \
      --hidden-dim <hidden_dim> \
      --dataset-configs <dataset_name:config_name> ... \
      --layer <layer_1> ... \
      --in-dir <input_directory> \
      --in-path <input_path> \
      --out-dir <output_directory> \
      --out-path <output_path>
   ```

4. Identify Language-Specific Features

   ```bash
   python identify.py \
       --model <model_name> \
       --sae-model <sae_model_name> \
       --layer <layer_1> ... \
       --dataset-configs <dataset_name_1:config_name_1:split_name_1> ... \
       --in-dir <input_directory> \
       --in-path <input_path> \
       --out-dir <output_directory> \
       --out-path <output_path> \
       --out-filename <output_filename> \
       --top <top_n_features> \
       --topk-threshold-ratio <topk_threshold_ratio> \
       --example-rate <example_rate> \
       --entropy-threshold <entropy_threshold> \
       --algorithm "sae-lape" \
       [ --lang-shared ] \
       --shared-count 2 \
       [ --top-per-layer ] \
       [ --lang-specific ] \
       [ --top-by-frequency ]
   ```

   > Note: `--shared-count` is the number of shared languages. If `--lang-shared` is not specified, it will be ignored.

5. Interpret SAE Features

   ```bash
   python interpret.py meta-llama/Llama-3.2-1B EleutherAI/sae-Llama-3.2-1B-131k \
      --hookpoints layers.0.mlp layers.1.mlp layers.2.mlp layers.3.mlp layers.4.mlp layers.5.mlp layers.6.mlp layers.7.mlp layers.8.mlp layers.9.mlp layers.10.mlp layers.11.mlp layers.12.mlp layers.13.mlp layers.14.mlp layers.15.mlp \
      --explainer_model openai/gpt-4.1 \
      --explainer_model_max_len 5120 \
      --explainer_provider openrouter \
      --name interpret_sae_features \
      --seed 42 \
      --num_examples_per_scorer_prompt 5 \
      --dataset_configs facebook/xnli:{en,de,fr,hi,es,th,bg,ru,tr,vi}:train google-research-datasets/paws-x:{en,de,fr,es}:train openlanguagedata/flores_plus:{eng_Latn,deu_Latn,fra_Latn,ita_Latn,por_Latn,hin_Deva,spa_Latn,tha_Thai,bul_Cyrl,rus_Cyrl,tur_Latn,vie_Latn,jpn_Jpan,kor_Hang,cmn_Hans}:dev \
      --dataset_start 0 --dataset_end 1000 \
      --dataset_repo "" \
      --dataset_split "" \
      --dataset_name "" \
      --batch_size 32 \
      --cache_ctx_len 256 \
      --n_tokens 10_000_000 \
      --n_splits 5 \
      --faiss_embedding_cache_enabled false \
      --example_ctx_len 32 \
      --min_examples 200 \
      --n_non_activating 50 \
      --non_activating_source random \
      --center_examples true \
      --n_examples_train 40 \
      --n_examples_test 50 \
      --n_quantiles 10 \
      --train_type top \
      --test_type quantiles \
      --lape_result_path sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt \
      --env local \
      --verbose false \
      --log_results true \
      --populate_cache true \
      --scoring true
   ```

6. Train SAE

   `train_sae.py` is a script to train SAEs, which is based on the `sparsify` code.

   Examples:

   ```bash
   python train_sae.py meta-llama/Llama-3.2-1B EleutherAI/rpj-v2-sample \
        --hf_token {HF_TOKEN} \
        --expansion_factor 64 \
        --normalize_decoder True \
        --k 32 \
        --multi_topk False \
        --skip_connection False \
        --batch_size 4 \
        --grad_acc_steps 2 \
        --micro_acc_steps 1 \
        --lr_warmup_steps 1000 \
        --auxk_alpha 0.0 \
        --dead_feature_threshold 10000000 \
        --hookpoints layers.0.mlp \
        --init_seeds 0 \
        --layer_stride 1 \
        --transcode False \
        --distribute_modules False \
        --save_every 1000 \
        --log_to_wandb True \
        --run_name 'sae' \
        --wandb_log_frequency 1 \
        --split train \
        --ctx_len 2048 \
        --load_in_8bit False \
        --resume False \
        --text_column raw_content \
        --shuffle_seed 42 \
        --data_preprocessing_num_proc 112
   ```

   > Note: This script requires a huge amount of memory, so it is recommended to pretokenized the dataset first using the `train_sae_pretokenize.py` script.

### MLP Activations

1. Count MLP Activations

   ```bash
   python activations_count.py <model_name> \
      --dataset-configs <dataset_name:config_name:split_name:start:end> ... \
      --layer <layer_1> ... \
      --start <row_number> --end <row_number> \
      --out-dir <output_directory> \
      --out-path <output_path> \
      --hidden-dim <hidden_dim> 
   ```

2. Identify Language-Specific Neurons

   ```bash
   python identify.py \
       --model <model_name> \
       --layer <layer_1> ... \
       --dataset-configs <dataset_name_1:config_name_1:split_name_1> ... \
       --in-dir <input_directory> \
       --in-path <input_path> \
       --out-dir <output_directory> \
       --out-path <output_path> \
       --out-filename <output_filename> \
       --algorithm "lape" \
   ```

### Experiments

1. Text Generation Experiment

   ```bash
   python text_generation.py \
      --model <model_name> \
      --layer <layer_name> ... \
      --in-dir <input_directory> \
      --in-path <input_path> \
      --out-dir <output_directory> \
      --out-path <output_path> \
      --multiplier <multiplier> \
      --max-new-token <max_new_token> \
      --times <times> \
      --seed <seed> \
      --lang <lang_1> ... \
      --prompt <prompt> \
      [ --last-token-only ] 
   ```

   > Note:
   > `--last-token-only` option is used to generate text with intervention on the last token only. Without this option, the text will be generated with intervention on all tokens.

2. Compute perplexity

   ```bash
   python ppl.py <model_name> <dataset_name> \
      --split <dataset_split_name> \
      --local-path <local_path_1> ... \
      --lang <lang_1> ... \
      --start <row_number> --end <row_number> \
      --out-dir <output_directory> \
      --out-path <output_path> \
      --layer <layer_1> ... \
      --intervention-type <intervention_type> \
      --intervention-lang <intervention_lang> \
      --multiplier <multiplier> \
      --value <value> \
      --neuron-intervention-method <type> \
      --lape-result-path <lape_result_path> \
      --lape-value-type <lape_value_type>
   ```

3. Classification Experiment

   ```bash
   python classify.py MartinThoma/wili_2018 \
      --model meta-llama/Llama-3.2-1B \
      --split test \
      --lang eng deu fra ita por hin spa tha bul rus tur vie jpn kor zho \
      --layer model.layers.{0..15}.mlp \
      --start 0 --end 500 \
      --sae-model EleutherAI/sae-Llama-3.2-1B-131k \
      --batch 500 \
      --lape-result-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt' \
      --classifier-type sae-count

   python classify.py MartinThoma/wili_2018 \
      --model meta-llama/Llama-3.2-1B \
      --split test \
      --lang eng deu fra ita por hin spa tha bul rus tur vie jpn kor zho \
      --layer model.layers.{0..15}.mlp.act_fn \
      --start 0 --end 500 \
      --lape-result-path 'mlp_acts_specific/meta-llama/Llama-3.2-1B/lape_neuron.pt' \
      --classifier-type neuron-count

   python classify.py MartinThoma/wili_2018 \
      --split test \
      --lang eng deu fra ita por hin spa tha bul rus tur vie jpn kor zho \
      --start 0 --end 500 \
      --classifier-type fasttext
   ```

4. Check language-specific features activations

   ```bash
   python check_active_features.py \
      --model meta-llama/Llama-3.2-1B \
      --layer model.layers.{0..15}.mlp \
      --sae-model EleutherAI/sae-Llama-3.2-1B-131k \
      --batch 500 \
      --lape-result-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt' \
      --classifier-type count \
      --text "Hamilton has been the primary songwriter, guitarist, and vocalist for Brothers Past, as well as co-producer for all of their recorded releases." "Olga Alexandrowna Girja (russisch Ольга Александровна Гиря; * 4. Juni 1991 in Langepas) ist eine russische Schachspielerin und seit 2009 Großmeister der Frauen (WGM)."
   ```

## Citation

```bibtex
@misc{andrylie2025sparseautoencoderscapturelanguagespecific,
      title={Sparse Autoencoders Can Capture Language-Specific Concepts Across Diverse Languages}, 
      author={Lyzander Marciano Andrylie and Inaya Rahmanisa and Mahardika Krisna Ihsani and Alfan Farizki Wicaksono and Haryo Akbarianto Wibowo and Alham Fikri Aji},
      year={2025},
      eprint={2507.11230},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.11230}, 
}
```
