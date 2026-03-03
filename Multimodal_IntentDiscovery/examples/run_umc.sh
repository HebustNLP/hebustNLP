#!/usr/bin/bash
# 2.4、3
# 0.1
# temp_umc
for seed in 0 1 3 4
do
    for multimodal_method in 'umc'
    do
        for method in 'umc'
        do 
            for text_backbone in 'bert-base-uncased'
            do
                for dataset in IEMOCAP-DA          #'MELD-DA' #'MIntRec' # 'MELD-DA' 'IEMOCAP-DA'
                do
                    python run.py \
                    --dataset $dataset \
                    --data_path '/home/sharing/disk1/Datasets' \
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
                    --results_file_name "results_umc_IEMOCAP_2.8.csv" \
                    --output_path "outputs/pre_IEMOCAP_16_4/${dataset}"
                done
            done
        done
    done
done
