#!/usr/bin/bash


for seed in 0 1 2
do
    for multimodal_method in 'dlla'
    do
        for method in 'dlla'
        do 
            for text_backbone in 'bert-base-uncased'
            do
                for dataset in 'MIntRec' #'MIntRec' # 'MELD-DA' 'IEMOCAP-DA'
                do
                    python run.py \
                    --dataset $dataset \
                    --data_path '/root/autodl-tmp/autodl' \
                    --logger_name $method \
                    --multimodal_method $multimodal_method \
                    --method $method \
                    --tune \
                    --save_results \
                    --save_model \
                    --seed $seed \
                    --gpu_id '0' \
                    --video_feats_path 'swin_feats.pkl' \
                    --audio_feats_path 'wavlm_feats.pkl' \
                    --text_backbone $text_backbone \
                    --config_file_name ${method}_${dataset} \
                    --results_file_name "dlla_MIntRec.csv" \
                    --output_path "/root/autodl-tmp/MIntRec"
                done
            done
        done
    done
done
