# From tensorflow/models/research/
INPUT_TYPE=image_tensor

#PIPELINE_CONFIG_PATH={path to pipeline config file}
PIPELINE_CONFIG_PATH=configs/ssd_inception_v2_alet.config

#TRAINED_CKPT_PREFIX={path to model.ckpt}
TRAINED_CKPT_PREFIX=saved_checkpoints/model.ckpt-0

# OUTPUT DIR
EXPORT_DIR=./


python export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
