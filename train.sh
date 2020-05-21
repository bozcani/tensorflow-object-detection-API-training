# Output directory to saved model checkpoints.
MODEL_DIR=training/

#PIPELINE_CONFIG_PATH={path to pipeline config file}
PIPELINE_CONFIG_PATH=configs/ssd_inception_v2_alet.config


python model_main.py \
    --logtostderr \
    --model_dir=${MODEL_DIR}  \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH}
