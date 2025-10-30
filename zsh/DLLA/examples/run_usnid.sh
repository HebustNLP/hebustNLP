#!/usr/bin/bash

for seed in 0 1 2
do
    for multimodal_method in 'text'
    do
        for method in 'usnid'
        do 
            for text_backbone in 'bert-base-uncased'
            do
                python run.py \
                --dataset 'MELD-DA' \
                --logger_name $method \
                --multimodal_method $multimodal_method \
                --method $method\
                --train \
                --tune \
                --save_results \
                --save_model \
                --seed $seed \
                --gpu_id '0' \
                --video_feats_path 'swin_feats.pkl' \
                --audio_feats_path 'wavlm_feats.pkl' \
                --text_backbone $text_backbone \
                --config_file_name $method \
                --results_file_name "baseline/results_$method.csv" \
                --output_path '/root/autodl-tmp/model/usnid' \
                --data_path '/root/autodl-tmp/autodl'  
            done
        done
    done
done