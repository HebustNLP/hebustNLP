#!/usr/bin/bash

for seed in 0
do
    for multimodal_method in 'mcn'
    do
        for method in 'mcn'
        do 
            for text_backbone in 'bert-base-uncased'
            do
                python run.py \
                --dataset 'MIntRec' \
                --logger_name $method \
                --multimodal_method $multimodal_method \
                --method $method\
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
                --output_path '/root/autodl-tmp/model/mcn' \
                --data_path '/root/autodl-tmp/autodl'
            done
        done
    done
done