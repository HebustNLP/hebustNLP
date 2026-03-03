#!/usr/bin/bash

for seed in 0 1 2 3 4
do
    for multimodal_method in 'text'
    do
        for method in 'sccl'
        do 
            for text_backbone in 'bert-base-uncased'
            do
                python run.py \
                --dataset 'MIntRec' \
                --logger_name $method \
                --multimodal_method $multimodal_method \
                --method $method\
                --train \
                --tune \
                --save_results \
                --save_model \
                --seed $seed \
                --gpu_id '1' \
                --video_feats_path 'video_feats.pkl' \
                --audio_feats_path 'audio_feats.pkl' \
                --text_backbone $text_backbone \
                --config_file_name $method \
                --results_file_name "baseline/results_$method.csv" \
                --output_path '/home/sharing/disk1/disk1/zhoushihao/zhou/sccl' \
                --data_path '/home/sharing/disk1/Datasets'
            done
        done
    done
done