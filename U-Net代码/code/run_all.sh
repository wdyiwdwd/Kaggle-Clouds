#!/bin/bash


DEVICE_ID=0,1,2,3
TTA=4

MODEL_NAMES=(
  model_a_00
  model_a_02
  model_a_01
  )


##################################################################################
# inference all models
##################################################################################
for MODEL_NAME in ${MODEL_NAMES[@]}; do
  CONFIG=configs/$MODEL_NAME.yaml
  CHECKPOINT=checkpoints/$MODEL_NAME.pth
  OUTPUT_PATH=inference_results/$MODEL_NAME

  #CUDA_VISIBLE_DEVICES=$DEVICE_ID 
  python run.py inference with $CONFIG evaluation.batch_size=1 inference.output_path=$OUTPUT_PATH inference.split=dev checkpoint=$CHECKPOINT -f
  python run.py inference with $CONFIG evaluation.batch_size=1 inference.output_path=$OUTPUT_PATH inference.split=test_dev checkpoint=$CHECKPOINT -f
  python run.py inference with $CONFIG evaluation.batch_size=1 inference.output_path=$OUTPUT_PATH inference.split=test checkpoint=$CHECKPOINT -f
done



##################################################################################
# evaluate & inference 
##################################################################################
MODEL_NAMES=(
  model_a_00
  model_a_01
  model_a_02
  )

OUTPUTS=
for MODEL_NAME in ${MODEL_NAMES[@]}; do
  OUTPUT_PATH=inference_results/$MODEL_NAME
  OUTPUTS=$OUTPUTS,$OUTPUT_PATH
done

OUTPUTS=${OUTPUTS:1}

python tools/evaluate.py --input_dir $OUTPUTS
python tools/make_submission.py --input_dir $OUTPUTS --output=submissions/reproduce.csv
