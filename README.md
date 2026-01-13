

<img src="assets/logo.png" width="400">

# Beyond Hard Masks: Progressive Token Evolution for Diffusion Language

[![Paper](https://img.shields.io/badge/Paper-Arxiv%20Link-green)](https://arxiv.org/abs/2601.07351) [![Project](https://img.shields.io/badge/Project-Page-blue)](https://aim-uofa.github.io/EvoTokenDLM) [![Code](https://img.shields.io/badge/Code-GitHub-orange)](https://github.com/aim-uofa/EvoTokenDLM) [![LoRA](https://img.shields.io/badge/Weights-EvoToken_LoRA-yellow)](https://huggingface.co/zhongzero/EvoToken_LLaDA_Instruct_8B_Lora) [![License](https://img.shields.io/badge/License-BSD%202--clause-lightgrey)](https://opensource.org/license/bsd-2-clause)

## 📣 News

- [2025-01-13] Code of EvoToken-DLM Released!
- [2025-01-12] Paper Released!


## 🚀 Overview

We propose **EvoToken-DLM**, a novel diffusion-based language modeling approach that replaces hard binary masks with evolving soft token distributions.

<img src="assets/overview.png"  width="1000">

## 📖 Description

Diffusion Language Models (DLMs) offer a promising alternative for language modeling by enabling parallel decoding through iterative refinement.
However, most DLMs rely on hard binary masking and discrete token assignments, which hinder the revision of early decisions and underutilize intermediate probabilistic representations. We propose **EvoToken-DLM**, a novel diffusion-based language modeling approach that replaces hard binary masks with evolving soft token distributions. EvoToken-DLM enables a progressive transition from masked states to discrete outputs, supporting revisable decoding. To effectively support this evolution, we introduce **continuous trajectory supervision**, which aligns training objectives with iterative probabilistic updates. Extensive experiments across multiple benchmarks show that EvoToken-DLM consistently achieves superior performance, outperforming strong diffusion-based and masked DLM baselines.



## ⚙️ Getting Started

### Environment Setup

To setup the environment, run:
```
conda env create -f env.yaml
conda activate evotoken
```



### Inference

Progressive inference with evolving soft token distributions.

```
python generate.py  --model_path GSAI-ML/LLaDA-8B-Instruct \
    --checkpoint_path zhongzero/EvoToken_LLaDA_Instruct_8B_Lora \
    --prompt "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?" \
    --k_soft 3 \
    --alpha_soft_mask 0.7
```

The parameters `--k_soft` and `--alpha_soft_mask` are adjustable hyperparameters; for specific details, please refer to the paper.

For your convenience, the generation process is encapsulated in `run.sh`, which can be executed directly.



### Evaluation

Evaluation on Countdown, GSM8K, MATH500 and SVAMP datasets.

```bash
bash eval/start_run.sh
```

Run `eval/get_acc.py` after the evaluation is done.



### Training

Training using continuous trajectory supervision.

```
bash train/train_CTS.sh
```





## 🎫 License

For academic use, this project is licensed under [the 2-clause BSD License](https://opensource.org/license/bsd-2-clause). 
For commercial use, please contact [Chunhua Shen](mailto:chhshen@gmail.com).

## 🖊️ Citation

If you find this work useful, please consider citing:

```bibtex 
@article{zhong2026beyond,
    title={Beyond Hard Masks: Progressive Token Evolution for Diffusion Language Models},
    author={Zhong, Linhao and Wu, Linyu and Fang, Bozhen and Feng, Tianjian and Jing, Chenchen and Wang, Wen and Zhang, Jiaheng and Chen, Hao and Shen, Chunhua},
    journal={arXiv preprint arXiv:2601.07351},
    year={2026}
}
```