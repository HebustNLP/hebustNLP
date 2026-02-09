#!/usr/bin bash
for seed in  2
do
    for dataset in  'clinc' #'banking' 'clinc' 'stackoverflow'
    do  
        for known_cls_ratio in  0.75  #0.5 0.25  
        do
            for cluster_num_factor in 1.0 #2.0 3.0 4.0
            do
                python run.py \
                --dataset $dataset \
                --method 'SRBS' \
                --setting 'semi_supervised' \
                --known_cls_ratio $known_cls_ratio \
                --seed $seed \
                --train \
                --tune \
                --cluster_num_factor $cluster_num_factor \
                --backbone 'bert_SRBS' \
                --config_file_name 'SRBS' \
                --gpu_id '0' \
                --results_file_name 'xiaopre_experiment.csv' \
                --save_results \
                --save_model \
                --output_dir '/root/autodl-tmp/xiaopre_model' 
            done
        done
    done
done