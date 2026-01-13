CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:$PYTHONPATH accelerate launch --config_file train/ddp_config.yaml --main_process_port 29500 --num_processes 1 train/sft_train_CTS.py \
                                    --grad_accum_steps 4 \
                                    --batch_size 2 \
                                    --continuous_K 4 \
                                    --num_epochs 60 \
                                    --max_length 1024 \
                                    --train_data simplescaling/s1K \
                                    --model_name "GSAI-ML/LLaDA-8B-Instruct/" \
                                    --output_dir "lora_results"