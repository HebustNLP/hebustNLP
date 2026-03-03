#!/usr/bin/bash


for seed in 0
do
    for multimodal_method in 'cabt'
    do
        for method in 'cabt'
        do 
            for text_backbone in 'bert-base-uncased'
            do
                for dataset in 'MIntRec' #'MIntRec' # 'MELD-DA' 'IEMOCAP-DA'
                do
                    python run.py \
                    --dataset $dataset \
                    --data_path '/root/autodl-tmp/datasets' \
                    --logger_name $method \
                    --multimodal_method $multimodal_method \
                    --method $method \
                    --train \
                    --tune \
                    --save_results \
                    --save_model \
                    --seed $seed \
                    --gpu_id '0' \
                    --video_feats_path 'swin_feats.pkl' \
                    --audio_feats_path 'wavlm_feats.pkl' \
                    --text_backbone $text_backbone \
                    --config_file_name ${method}_${dataset} \
                    --results_file_name "cabt_MIntRec.csv" \
                    --output_path "/root/autodl-tmp/outputs/${dataset}"
                done
            done
        done
    done
done
