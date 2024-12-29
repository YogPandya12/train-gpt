# Reproducing GPT-2

This project demonstrates the reproduction of GPT-2, a decoder-only transformer architecture designed for natural language processing tasks. Inspired by Andrey Karpathy's work, I implemented and partially pre-trained the model, overcoming resource and dataset constraints. Below is a detailed description of the process and findings.

## Model Initialization

1. **Custom Model Configuration**:
   - Token embeddings, position embeddings, hidden layers, final layer normalization, and the classifier were initialized.
   - The core block structure implemented included:
     - `ln1`, `ln2` for layer normalization.
     - `attn` (attention mechanism) for inter-token communication.
     - `mlp` (feed-forward layers) for token-wise transformations.

2. **MLP Implementation**:
   - Components like `c_fc`, `c_proj`, and GELU activation were initialized for smoothing linear transformations.

3. **Self-Attention Implementation**:
   - Multi-head attention initialized with `c_attn`, `c_head`, `c_proj`, and `c_embd`.
   - Query (Q), Key (K), and Value (V) matrices computed to determine token relationships.

4. **Pretrained Model Conversion**:
   - The `from_pretrained()` function was created to map weights from Hugging Face's GPT-2 model.
   - Tensor alignment ensured compatibility with my model's structure.

## Forward Pass and Sampling

1. **Logits Calculation**:
   - Forward pass implemented in the GPT class to return logits and loss.

2. **Sampling**:
   - Used tokenization via `tiktokens` for generating samples.
   - Tokens selected based on top-k probabilities, with compatibility across CUDA, MPS, or CPU devices.

## Training the Model

1. **Dataset Preparation**:
   - Initial testing conducted with the Shakespeare dataset.
   - Data loader designed to read text files, tokenize data, and create batches.

2. **Optimization Loop**:
   - Overfitted on a single batch to verify model training.
   - Mixed-precision training (torch.float32 and bfloat16) applied for speed optimization.
   - Used `torch.compile` to minimize Python overhead and kernel fusion delays.

3. **Efficient Attention and Computation**:
   - Flash Attention implemented for speed optimization.
   - Adjusted vocabulary size from 50,257 to 50,304 (a power of 2) for CUDA efficiency.

4. **Hyperparameter Tuning**:
   - AdamW optimizer with gradient clipping (global norm capped at 1.0).
   - Cosine decay learning rate scheduler with warmup steps based on 375M tokens.

5. **Distributed Training**:
   - Leveraged multiple GPUs with Distributed Data Parallel (DDP) for large workloads.
   - Data loader modified for distributed training.

## Dataset and Training Challenges

- The original GPT-2 dataset (WebText) was unavailable. The Fineweb dataset was used, sampling 10B tokens for training.
- Due to storage constraints, dataset shards were streamed, pretokenized, and deleted dynamically after use.
- Training estimates for 10B tokens were derived based on the Shakespeare dataset's token/step ratio.

## Evaluation

- Evaluation was conducted using the HellaSwag dataset, which involves selecting the correct answer from multiple-choice options.
- Labels were tokenized, and options passed through the model. Initial training results showed promising performance.

## Results

- Loss consistently decreased from step 1 to step 380.
- Model checkpoints produced coherent answers for sample prompts.
- HellaSwag evaluation scores suggested that full training on 10B tokens could achieve results comparable to GPT-2.

## Screenshots and Logs

1. **Training Progress**:
   - [Step 240 Screenshot](step%20240.png): Intermediate training results.
   - [Step 380 Screenshot](step%20380.png): Loss and token statistics at a later stage of training.

2. **Logs**:
   - [Training Logs](log.txt): Includes sample outputs and HellaSwag evaluation results.

## References

- Implementation inspired by Andrey Karpathy's work.
- Dataset: Fineweb (10B tokens sampled).
- Evaluation Dataset: HellaSwag.

---

For more details, check the associated training scripts and the uploaded resources in this repository.
