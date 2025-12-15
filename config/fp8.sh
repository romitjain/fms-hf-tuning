SFT_TRAINER_CONFIG_JSON_PATH=./config/sft.json \
accelerate launch --config_file ./config/fsdp_mp_fp8.yaml -m tuning.sft_trainer