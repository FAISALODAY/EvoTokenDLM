import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers import DefaultDataCollator
from transformers import DataCollatorWithPadding
import random
from tqdm import tqdm
import pickle
import torch.distributed as dist

@ torch.no_grad()
def process_soft_token(logits, x, confidence_x, mask_id=126336, k_soft = 3, alpha_soft_mask = 0.8, block_start=None, prompt_len=None):
    x0 = torch.argmax(logits, dim=-1) # b, l
    p = F.softmax(logits.to(torch.float64), dim=-1)
    x0_p = torch.squeeze(
        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    confidence_x0 = x0_p.clone()
    
    decode_index = (x != mask_id)
    if prompt_len is not None:
        decode_index[:, :prompt_len] = 0
        
    decode_index[:, :block_start] = 0
    better_decode_index = decode_index & (confidence_x0 > confidence_x)
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
    
    return soft_token.to(torch.long), prob.to(logits.dtype)

class dLLMTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Absorbing state diffusion loss computation
        """
        input_ids, ground_truth, prompt_lengths, start_block, end_block, next_unmask_positions, t = (
            inputs.pop("input_ids"),
            inputs.pop("GT"),
            inputs.pop("prompt_lengths"),
            inputs.pop("start_block"),
            inputs.pop("end_block"),
            inputs.pop("next_unmask_positions"),
            inputs.pop("t"),
        )
        K = next_unmask_positions.shape[1]
        per_deNum = next_unmask_positions.shape[2]
        B, N = input_ids.shape
        
        soft_token = None
        prob = None
        loss = 0.0
        confidence_x = torch.zeros_like(input_ids, dtype=torch.float64)
        for k in range(K):
            labels = ground_truth.clone()
            unsupervise_pos_begin = torch.arange(N, device=input_ids.device).unsqueeze(0) < start_block.unsqueeze(1)
            labels[unsupervise_pos_begin] = -100
            unsupervise_pos_end = torch.arange(N, device=input_ids.device).unsqueeze(0) > end_block.unsqueeze(1)
            labels[unsupervise_pos_end] = -100
            
            
            outputs = model(input_ids=input_ids, soft_token=soft_token, prob=prob)
            logits = outputs.logits
            unscaled_loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
            ).view(logits.shape[0], -1)
            current_loss = unscaled_loss.sum() / (labels != -100).sum()
            def statistics():
                num_mask_tokens = 0
                num_decode_tokens = 0
                for pos in range(start_block[0], end_block[0]+1):
                    if input_ids[0][pos] == 126336:
                        num_mask_tokens += 1
                    else:
                        num_decode_tokens += 1
                mask_loss = unscaled_loss[0][start_block[0]:end_block[0]+1][input_ids[0][start_block[0]:end_block[0]+1]==126336]
                decode_loss = unscaled_loss[0][start_block[0]:end_block[0]+1][input_ids[0][start_block[0]:end_block[0]+1]!=126336]
                if mask_loss.numel() > 0:
                    mask_loss = mask_loss.mean()
                else:
                    mask_loss = torch.tensor(0.0)
                if decode_loss.numel() > 0:
                    decode_loss = decode_loss.mean()
                else:
                    decode_loss = torch.tensor(0.0)
                print(f"Mask token loss: {mask_loss.item():.4f}, Decode token loss: {decode_loss.item():.4f}")
            
            # statistics()
            if (self.state.global_step + 1) % self.args.logging_steps == 0:
                self.log({f"loss in stage_{k}": current_loss.item()})
            loss += current_loss

            predicted_tokens = torch.argmax(logits, dim=-1)
            for b in range(B):
                for d in range(per_deNum):
                    mask_pos = next_unmask_positions[b, k, d]
                    input_ids[b, mask_pos] = predicted_tokens[b, mask_pos]
            soft_token_list = []
            prob_list = []
            for b in range(B):
                alpha_soft_mask = random.uniform(0.5, 1.0)
                soft_token_b, prob_b = process_soft_token(logits[b:b+1], input_ids[b:b+1], confidence_x[b:b+1], mask_id=126336, 
                                k_soft = 3, alpha_soft_mask = alpha_soft_mask, block_start=start_block[b], prompt_len=prompt_lengths[b])
                soft_token_list.append(soft_token_b)
                prob_list.append(prob_b)
            soft_token = torch.cat(soft_token_list, dim=0)
            prob = torch.cat(prob_list, dim=0)
            input_ids = input_ids.detach()
            soft_token = soft_token.detach()
            prob = prob.detach()
            assert not input_ids.requires_grad
            assert not soft_token.requires_grad
            assert not prob.requires_grad
        loss = loss / K
        return loss if not return_outputs else (loss, outputs)


class dLLMSFTDataset(torch.utils.data.Dataset):
    """
    Similar to AR datasets, except in inference, we keep the timsteps fixed
    """

    def __init__(self, data, tokenizer, max_length, eval=False):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eval = eval

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out = self.data[idx]
        return out


class dLLMDataCollator(DefaultDataCollator):
    """
    Adds the forward noising process to the batch.
    Modify forward_process to change the noise schedule
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mask_token_id = kwargs["tokenizer"].mask_token_id
        self.tokenizer = kwargs["tokenizer"]
        if "max_length" in kwargs:
            self.max_length = kwargs["max_length"]
        if kwargs["tokenizer"].mask_token_id is None:
            assert (
                "mask_token_id" in kwargs
            ), "For dLLM models, pass a mask_token_id or set it equal to tokenizer.mask_token_id"
            self.mask_token_id = kwargs["mask_token_id"]
        
        self.K = kwargs.get("K", 4)
        self.per_deNums = kwargs.get("per_deNums", [1, 2, 4, 8])
        self.block_sizes = kwargs.get("block_sizes", [32, 64, 128, 256, 512])

        self.pad_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, padding="longest", return_tensors="pt"
        )

    def forward_process(self, batch, eps=1e-3):
        input_ids = batch["input_ids"]
        prompt_lengths = batch["prompt_lengths"]
        B, N = input_ids.shape
        
        per_deNum = random.choice(self.per_deNums)
        L = random.choice(self.block_sizes)
        if L > N - prompt_lengths.max().item():
            L = N - prompt_lengths.max().item()
        if self.K * per_deNum >= L:
            L = self.K * per_deNum + 1
        assert L <= N, "Block size larger than sequence length!"
        
        start_block = torch.zeros((B,), dtype=torch.long, device=input_ids.device)
        end_block = torch.zeros((B,), dtype=torch.long, device=input_ids.device)
        mask_indices = torch.zeros((B, N), dtype=torch.bool, device=input_ids.device)
        next_unmask_positions = torch.zeros((B, self.K, per_deNum), dtype=torch.long, device=input_ids.device)
        
        for b in range(B):
            start_min = prompt_lengths[b].item()
            start_max = N - L
            if start_min >= start_max:
                start_block[b] = start_max
            else:
                start_block[b] = random.randint(start_min, start_max)
            end_block[b] = start_block[b] + L - 1
            P = torch.randperm(L)
            mask_num = random.randint(0, L - self.K * per_deNum)
            mask_indices[b, :start_block[b]] = False
            mask_indices[b, end_block[b]+1:] = True
            for i in range(mask_num, L):
                mask_pos = start_block[b] + P[i].item()
                mask_indices[b, mask_pos] = True

            for k in range(self.K):
                for d in range(per_deNum):
                    mask_pos = start_block[b] + P[mask_num + k * per_deNum + d].item()
                    next_unmask_positions[b, k, d] = mask_pos
        
        t = None

        noisy_batch_init = torch.where(mask_indices, self.mask_token_id, input_ids)
        return noisy_batch_init, t, mask_indices, start_block, end_block, next_unmask_positions

    def __call__(self, batch):
        batch = self.pad_collator(batch)
        batch["attention_mask"] = torch.ones_like(batch["input_ids"], dtype=torch.long, device=batch["input_ids"].device)
        if random.random() < 0.5:
            B, N = batch["input_ids"].shape
            extra_len = random.randint(1, 10)
            pad_tensor = torch.full((B, extra_len), self.tokenizer.pad_token_id, dtype=torch.long, device=batch["input_ids"].device)
            batch["input_ids"] = torch.cat((batch["input_ids"], pad_tensor), dim=1)
            pad_attention = torch.ones((B, extra_len), dtype=torch.long, device=batch["attention_mask"].device)
            batch["attention_mask"] = torch.cat((batch["attention_mask"], pad_attention), dim=1)

        batch["GT"] = batch["input_ids"].clone()
        noisy_batch_init, batch["t"], mask_indices, start_block, end_block, next_unmask_positions = self.forward_process(batch)
        batch["input_ids"] = noisy_batch_init.long()
        batch["start_block"] = start_block
        batch["end_block"] = end_block
        batch["next_unmask_positions"] = next_unmask_positions
        return batch


SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>
"""


def preprocess_dataset(data, train_data_path, tokenizer, max_length, max_dataNum=2000, test_split=0.01):
    min_response_length=32
    
    data = [item for dataset in data for item in dataset]
    
    print(f"Original dataset length: {len(data)}")
    if "simplescaling/s1K" in train_data_path:
        print("Using simplescaling/s1K dataset preprocessing rules.")
        data = [item for item in data if item.get('cot_type', '') == 'math']
        print(f"Filtered dataset length (cot_type='math'): {len(data)}")
        filter_data = []
        for i in range(len(data)):
            answer_start = data[i]['attempt'].rfind('\\boxed{')
            answer_end = data[i]['attempt'].rfind('}')
            if answer_start != -1 and answer_end != -1 and answer_end > answer_start:
                answer_content = data[i]['attempt'][answer_start + len('\\boxed{'):answer_end]
                trajectory = f"<reasoning>{data[i]['attempt']}</reasoning>\n<answer>\\boxed{{{answer_content}}}</answer>"
            else:
                continue
            data[i]['answer'] = trajectory
            keys_to_remove = [key for key in data[i].keys() if key not in ['question', 'answer']]
            for key in keys_to_remove:
                del data[i][key]
            filter_data.append(data[i])
        data = filter_data
        print(f"Final dataset length after filtering invalid answers: {len(data)}")
    elif "openai/gsm8k" in train_data_path:
        print("Using openai/gsm8k dataset preprocessing rules.")
        filter_data = []
        for i in range(len(data)):
            answer_start = data[i]['answer'].rfind('#### ')
            if answer_start != -1:
                answer_content = data[i]['answer'][answer_start + len('#### '):].strip()
                trajectory = f"<reasoning>{data[i]['answer'][:answer_start]}</reasoning>\n<answer>\\boxed{{{answer_content}}}</answer>"
            else:
                continue
            data[i]['answer'] = trajectory
            keys_to_remove = [key for key in data[i].keys() if key not in ['question', 'answer']]
            for key in keys_to_remove:
                del data[i][key]
            filter_data.append(data[i])
        data = filter_data
        print(f"Final dataset length after filtering invalid answers: {len(data)}")
    elif "EleutherAI/hendrycks_math" in train_data_path:
        print("Using EleutherAI/hendrycks_math dataset preprocessing rules.")
        filter_data = []
        for i in range(len(data)):
            answer_start = data[i]['solution'].rfind('\\boxed{')
            answer_end = data[i]['solution'].rfind('}')
            if answer_start != -1 and answer_end != -1 and answer_end > answer_start:
                answer_content = data[i]['solution'][answer_start + len('\\boxed{'):answer_end]
                trajectory = f"<reasoning>{data[i]['solution']}</reasoning>\n<answer>\\boxed{{{answer_content}}}</answer>"
            data[i]['question'] = data[i]['problem']
            data[i]['answer'] = trajectory
            keys_to_remove = [key for key in data[i].keys() if key not in ['question', 'answer']]
            for key in keys_to_remove:
                del data[i][key]
            filter_data.append(data[i])
        data = filter_data
        print(f"Final dataset length after filtering invalid answers: {len(data)}")
    else:
        raise NotImplementedError(f"Preprocessing rules for this dataset in '{train_data_path}' are not implemented.")
    
    preprocessed_data = []
    for i in tqdm(range(len(data)), desc="Preprocessing dataset"):
        question = SYSTEM_PROMPT + "\n\n" + data[i]["question"]
        
        answer = data[i]["answer"]
        
        prompt = [{"role": "user", "content": question}]
        response = [{"role": "assistant", "content": answer}]
        inputs = tokenizer.apply_chat_template(prompt + response, tokenize=False)
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False) + "\n"
        
        length = tokenizer(inputs, return_tensors="pt", truncation=False).input_ids.shape[1]
        if length > max_length:
            continue
        
        tokenized_input = tokenizer(
            inputs, return_tensors="pt", truncation=True, padding=False
        ).input_ids.squeeze(0)
        
        
        num_tokens = tokenized_input.shape[0]
        tokenized_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        prompt_lengths = tokenized_prompt.attention_mask.sum(-1)
        
        if length - prompt_lengths.item() < min_response_length:
            continue
        preprocessed_data.append(
            {
                "input_ids": tokenized_input,
                "prompt_lengths": prompt_lengths,
            }
        )
    print(f"Preprocessing completed. Total samples: {len(preprocessed_data)}")
    random.shuffle(preprocessed_data)
    if len(preprocessed_data) > max_dataNum:
        preprocessed_data = preprocessed_data[:max_dataNum]
        print(f"Truncated preprocessed data number to max_dataNum: {max_dataNum}")
    
    test_data = preprocessed_data[: int(len(preprocessed_data) * test_split)]
    train_data = preprocessed_data[int(len(preprocessed_data) * test_split) :]
    return train_data, test_data
