#!/bin/zsh

model_name_1=$MODEL_NAME_1
model_name_2=$MODEL_NAME_2
input_dir="dataset"
output_dir="generated-data"
batch_size=4
num_gpus=1
force_sequential=false  # Set to true if multiprocessing causes issues

# Create output directory if it doesn't exist
mkdir -p $output_dir

# Run the parallel processing script
python GranularTokenScoring.py \
  --model_name_1 $model_name_1 \
  --model_name_2 $model_name_2 \
  --model1_template $model1_template \
  --model2_template $model2_template \
  --input_dir $input_dir \
  --output_dir $output_dir \
  --batch_size $batch_size \
  --num_gpus $num_gpus \
  $(if $force_sequential; then echo "--force_sequential"; fi) 