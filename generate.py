import argparse
import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from model.modeling_llada import LLaDAModelLM


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        threshold: Confidence threshold for remasking. If specified, only tokens with confidence below this threshold will be remasked.
    '''
    print("generate call")
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            x[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    return x


@ torch.no_grad()
def generate_soft_token(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None,  k_soft=3,  alpha_soft_mask=0.8):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        threshold: Confidence threshold for remasking. If specified, only tokens with confidence below this threshold will be remasked.
        k_soft: Keep top k probability for each pure soft token.
        alpha_soft_mask: The mixing ratio for mask token in mask-aware soft token.
    '''
    print("generate_soft_token call")
    print(f"k_soft: {k_soft}, alpha_soft_mask: {alpha_soft_mask}")
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks
    
    soft_token = None
    prob = None
    
    confidence_x = torch.zeros_like(x, dtype=torch.float64)

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = block_start + block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            mask_index = (x == mask_id)
            logits = model(x, soft_token=soft_token, prob=prob).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            x0, transfer_index, soft_token, prob = get_transfer_index_soft_token(logits, temperature, remasking, mask_index, mask_id, x, confidence_x, num_transfer_tokens[:, i] if threshold is None else None, threshold, 
                k_soft=k_soft, alpha_soft_mask=alpha_soft_mask,
                block_start=block_start, prompt_len=prompt.shape[1])
            i += 1
            if (x[:, block_start: block_end] == mask_id).sum() == 0:
                break
    return x


def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None, mask_id=126336):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    logits_with_noise[:, :, mask_id] = -torch.inf
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index


def get_transfer_index_soft_token(logits, temperature, remasking, mask_index, mask_id, x, confidence_x, num_transfer_tokens, threshold=None, k_soft = 3, alpha_soft_mask = 0.8, block_start=None, prompt_len=None):
    from generate import (
        add_gumbel_noise,
    )
    
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    logits_with_noise[:, :, mask_id] = -torch.inf
    
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    confidence_x0 = x0_p.clone()
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False

    x[transfer_index] = x0[transfer_index]
    confidence_x[transfer_index] = confidence_x0[transfer_index]
    
    decode_index = (x != mask_id)
    if prompt_len is not None:
        decode_index[:, :prompt_len] = 0
        
    decode_index[:, :block_start] = 0
    better_decode_index = decode_index & (confidence_x0 > confidence_x) & (x0 != mask_id)
    x[better_decode_index] = x0[better_decode_index]
    confidence_x[better_decode_index] = confidence_x0[better_decode_index]

    prob, soft_token = torch.topk(p, k=k_soft, dim=-1)  # both (B, L, k_soft)
    
    ori_prob_mask = prob.sum(dim=-1, keepdim=True) / (1.0 - alpha_soft_mask) * alpha_soft_mask  # (B, L, 1)
    ori_prob_decode = torch.zeros_like(ori_prob_mask)
    ori_prob = torch.where((x == mask_id).unsqueeze(-1), ori_prob_mask, ori_prob_decode)
    
    ori_token = x.unsqueeze(-1)
    prob = torch.cat((prob, ori_prob), dim=-1)
    soft_token = torch.cat((soft_token, ori_token), dim=-1)
    
    prob = prob / prob.sum(dim=-1, keepdim=True)
    
    prob[:, :block_start, :] = 0.0
    prob[:, :block_start, -1] = 1.0
    
    if prompt_len is not None:
        prob[:, :prompt_len, :] = 0.0
        prob[:, :prompt_len, -1] = 1.0

    return x0, transfer_index, soft_token.to(torch.long), prob.to(logits.dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--checkpoint_path", type=str, default="zhongzero/EvoToken_LLaDA_Instruct_8B_Lora")
    parser.add_argument("--prompt", type=str, default="Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?")
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--k_soft", type=int, default=3)
    parser.add_argument("--alpha_soft_mask", type=float, default=0.7)
    args = parser.parse_args()
    
    device = args.device
    model = LLaDAModelLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if args.checkpoint_path is not None:
        model = PeftModel.from_pretrained(model, args.checkpoint_path).eval()

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": args.prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    # out = generate(model, input_ids, steps=args.diffusion_steps, gen_length=args.gen_length, block_length=args.block_length, temperature=args.temperature, remasking='low_confidence')
    out = generate_soft_token(model, input_ids, steps=args.diffusion_steps, gen_length=args.gen_length, block_length=args.block_length, temperature=args.temperature, remasking='low_confidence', k_soft=args.k_soft, alpha_soft_mask=args.alpha_soft_mask)
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])

if __name__ == '__main__':
    main()
