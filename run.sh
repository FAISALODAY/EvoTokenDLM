python generate.py  --model_path GSAI-ML/LLaDA-8B-Instruct \
                    --checkpoint_path zhongzero/EvoToken_LLaDA_Instruct_8B_Lora \
                    --prompt "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?" \
                    --k_soft 3 \
                    --alpha_soft_mask 0.7