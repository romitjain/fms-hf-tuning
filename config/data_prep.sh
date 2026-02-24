SFT_TRAINER_CONFIG_JSON_PATH=./config/data_prep.json \
accelerate launch --config_file ./config/fsdp_mp_bf16.yaml -m tuning.sft_trainer