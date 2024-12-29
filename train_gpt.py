# basic imports that we need

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from hellaswag import render_example, iterate_examples

# compute attention over tokens in a sequence while respecting causality which ensures that predictions for a given token only depend on previous tokens in the sequence

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y
    
# It defines a feed-forward network (FFN) used as a component. It operates on each token's representation independently after attention computations.

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

"""Each block combines self-attention and a feed-forward neural network (MLP) with residual connections and layer normalization. This architecture enables the model to learn complex relationships in sequential data effectively."""

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
"""This class is a configuration structure that defines the hyperparameters for the our GPT model. It encapsulates key parameters needed to initialize the model, specifying its size, capacity, and token processing capabilities."""

@dataclass
class GPTConfig:
    block_size: int = 1024        # max sequence length
    vocab_size: int = 50257       # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12             # number of layers
    n_head: int = 12              # number of heads
    n_embd: int = 768             # embedding dimension

"""This implementation defines a custom GPT model. It combines key components like embedding layers, Transformer blocks, and a language modeling head, while providing flexibility for weight initialization, optimizer configuration, and importing pretrained weights from HuggingFace's Transformers library."""

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * module.NANOGPT_SCALE_INIT) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)       # shape (T)
        pos_emb = self.transformer.wpe(pos)                                 # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)                                 # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]

        config_args['vocab_size'] = 50257     # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024      # always 1024 for GPT model checkpoints

        # create a from scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]     # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]            # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        """
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that in 2D will be weight decayed, otherwise no.
        # all weight tensors in matmuls + embeddings decay, all biases and layernorms dont.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        # fused Adam is used to speed up the training process
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), fused=use_fused, eps=1e-8)
        return optimizer

# used to check that all the initialization works perfectly
# model= GPT.from_pretrained('gpt2')        
# print('didnt crash yay!')

# to autodetect the device for training
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f'using device: {device}')

"""The dataset used for training the model is Fineweb10B. This class is used to download a shard of the dataset and load it for training the model. In the original training flow by Andrej Karpathy, the entire dataset was downloaded at once. I modified this approach due to a lack of storage resources, enabling the dataset to be downloaded in shards as training progresses and deleting each shard after it is processed."""

import os
import threading
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
import tiktoken

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, data_root, remote_name, shard_size):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        self.data_root = data_root
        self.remote_name = remote_name
        self.shard_size = shard_size
        self.lock = threading.Lock()
        os.makedirs(data_root, exist_ok=True)

        # Initialize dataset and tokenizer
        self.dataset = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", streaming=True)
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.eot = self.tokenizer._special_tokens['<|endoftext|>']  # End-of-text token

        # Initialize shard handling
        self.current_shard = None
        self.current_shard_idx = -1
        self.current_position = 0
        self.tokens = None

    # tokenize the text
    def tokenize(self, doc):
        tokens = [self.eot]
        tokens.extend(self.tokenizer.encode_ordinary(doc["text"]))
        tokens_np = np.array(tokens, dtype=np.uint16)
        return tokens_np

    def download_shard(self, shard_index):
        # Calculate which shard to download based on the process rank
        shard_index = shard_index + self.process_rank
        filename = os.path.join(self.data_root, f"edufineweb_{self.split}_{shard_index:06d}.npy")
        
        if os.path.exists(filename):
            return filename
        
        # Download and process the shard as before
        shard_tokens = []
        token_count = 0
        progress_bar = tqdm(total=self.shard_size, unit="tokens", desc=f"Shard {shard_index}")
        
        # We use a flag to stop once we fill the shard size
        for doc in self.dataset:
            tokens = self.tokenize(doc)
            if token_count + len(tokens) > self.shard_size:

                # Handle the remaining space in the current shard
                remaining_space = self.shard_size - token_count
                if remaining_space > 0:
                    shard_tokens.append(tokens[:remaining_space])  # Add only the remaining part
                    token_count += remaining_space
                    break  # Stop after the shard is filled
            else:
                shard_tokens.append(tokens)
                token_count += len(tokens)
                progress_bar.update(len(tokens))
        
        # If we haven't filled the shard to the required size, continue until it's filled
        while token_count < self.shard_size:
            for doc in self.dataset:
                tokens = self.tokenize(doc)
                remaining_space = self.shard_size - token_count
                if remaining_space > 0:
                    shard_tokens.append(tokens[:remaining_space])  # Add only the remaining part
                    token_count += remaining_space
                    break  # Stop once the shard is filled
                else:
                    shard_tokens.append(tokens)
                    token_count += len(tokens)
                    progress_bar.update(len(tokens))
            
            if token_count >= self.shard_size:
                break

        progress_bar.close()

        # Ensure that the shard is completely filled and saved
        all_tokens_np = np.concatenate(shard_tokens)
        np.save(filename, all_tokens_np)
        return filename

    # load the downloaded shard
    def load_tokens(self, filename):
        npt = np.load(filename)
        return torch.tensor(npt, dtype=torch.long)

    def reset(self):
        self.current_shard_idx += 1

        # Ensure only one process (GPU) handles deleting and downloading
        with self.lock:
            # Only delete the shard if it exists and hasn't been deleted already
            if self.current_shard and os.path.exists(self.current_shard):
                try:
                    print(f"Deleting previous shard: {self.current_shard}")
                    os.remove(self.current_shard)
                except FileNotFoundError:
                    print(f"Warning: Shard {self.current_shard} not found for deletion.")
                self.current_shard = None  # Clear reference after deletion

            # Download and load the new shard
            self.current_shard = self.download_shard(self.current_shard_idx)
            self.tokens = self.load_tokens(self.current_shard)
            self.current_position = self.B * self.T * self.process_rank

            print(f"Loaded new shard: {self.current_shard}")

    def next_batch(self):
        B, T = self.B, self.T
    
        # Reset the shard 
        if self.tokens is None or self.current_position + B * T + 1 > len(self.tokens):
            self.reset()
    
        # Calculate the correct position for each process based on rank
        batch_start = self.current_position + B * T * self.process_rank
        batch_end = batch_start + B * T + 1
    
        # Check if the batch exceeds the shard length
        if batch_end > len(self.tokens):
            self.reset()
    
        buf = self.tokens[batch_start:batch_end]
        
        # Check if buf is empty
        if buf.size(0) == 0:
            print(f"Warning: Empty batch detected. Resetting shard. Current position: {self.current_position}")
            self.reset()
            return self.next_batch()  # Recursively call to get a valid batch
    
        x = buf[:-1].view(B, T)  # Inputs
        y = buf[1:].view(B, T)   # Targets
    
        self.current_position += B * T * self.num_processes
        return x, y
    
"""This function is used to evaluate the model during training at regular intervals. The dataset used for evaluation is HellaSwag."""

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()

    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)

    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)

    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask

    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

"""Using DDP to distribute training across multiple GPUs to speed up the process and fully utilize GPU resources."""

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# setup DDP (distributed data parallel)
# torchrun command sets env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    # use DDP atm demands CUDA, we set device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA FOR DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0    # this process will do logging, checkpointing etc.

else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    # attempt to autodetect device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f'using device: {device}')

import time
import inspect

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 524288   # 2^19 ~0.5M, in number of tokens
B = 32                      # micro batch size
T = 1024                    # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, 'make sure total_batch_size is divisible by B * T * ddp_world_size'
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f'total desired batch size: {total_batch_size}')
    print(f'=> calculated gradient accumulation steps: {grad_accum_steps}')

# defining the dataset I am going to use, here we are gonna use fineweb10B and defined each shard of size 100M 
data_root = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8)

# The Fineweb10B dataset does not have a train and validation split by default. The validation loader can be used if the entire dataset has been downloaded. However, since I am directly using shards of the dataset, I am not using the validation loader here.
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train', data_root=data_root, remote_name=remote_name, shard_size=shard_size)
# val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

# set precision
torch.set_float32_matmul_precision('high')

# create model
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)   

# It compiles the model into an optimized representation, which can significantly speed up training
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model    # always contain raw unwrapped model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 91
max_steps = 2500
def get_lr(it):
    if it < warmup_steps:
        return min_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# optimize
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)  # using custom fused adam optimizer 

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps -1)
    # once in a while evaluate our validation loss
    # if step % 100 == 0:
    #     model.eval()
    #     val_loader.reset()
    #     with torch.no_grad():
    #         val_loss_accum = 0.0
    #         val_loss_steps = 20
    #         for _ in range(val_loss_steps):
    #             x, y = val_loader.next_batch()
    #             x, y = x.to(device), y.to(device)
    #             with torch.autocast(device_type = device, dtype = torch.bfloat16):
    #                 logits, loss = model(x, y)
    #             loss = loss / val_loss_steps
    #             val_loss_accum += loss.detach()
    #     if ddp:
    #         dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    #     if master_process:
    #         print(f'validation loss: {val_loss_accum.item():.4f}')

    # once in a while evaluate hellaswag
    if (step % 25 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0

        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue

            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)

            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)

        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # once in a while generate from model
    # disable because torch.compile throws a scary error i cant solve rn
    # if you disable torch.compile, this code works fine
    if step>0 and step%20 == 0:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            with torch.no_grad:
                logits, loss = model(xgen)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, k=10, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=1)
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # training loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type = device, dtype = torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    if master_process:
            print(f"loss: {loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {loss_accum.item():.4f}\n")
            if step > 0 and (step % 100 == 0 or last_step):
              
                # writing model checkpoints so we can use it later
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'loss': loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)
                  
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # determine and set learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()