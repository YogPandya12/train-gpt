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
   - Components like `c_fc`, `c_proj`, and `GELU` activation were initialized for smoothing linear transformations.

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

## Training computation improvement

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
- It is preferable to download the dataset before starting the training to make the process faster.
- But due to storage limitations, I was unable to download the entire dataset. Instead, I streamed shards, pre-tokenized them, and dynamically deleted the used shards while downloading the new batch.
- Training steps must be estimated in such a way that the model trains on 10 billion tokens. The steps can be derived based on the token-to-step ratio of the Shakespeare dataset, which was used to verify the training pipeline.
- Also, the GPT-2 paper discusses the warmup learning rate over 375 million tokens. The steps for the warmup learning rate must be calculated in a similar way to compute 375M tokens as well.

## Evaluation

- Evaluation was conducted using the HellaSwag dataset, which involves selecting the correct answer from multiple-choice options.
- Labels were tokenized, and options passed through the model. Initial training results showed promising performance.

## Results

- Loss consistently decreased from step 1 to step 380.
- Model checkpoints produced coherent answers for sample prompts.
- HellaSwag evaluation scores suggested that full training on 10B tokens could achieve results comparable to GPT-2.

## Screenshots and Logs
### Step 40
![Step 40](images/step%2040.png)

### Step 240
![Step 240](images/step%20240.png)

### Step 380
![Step 380](images/step%20380.png)
- The above screenshots show how the loss decreases as the model is being trained. Also, the sample review during training demonstrate how the model improves as it continues to train.
  
**Logs**:
   - [Training Logs](log.txt): The log file is recorded to showcase how the model responds during its training. Although the results are not very good because the model was not trained completely due to a lack of resources, there is clear improvement over time. Additionally, the code successfully streams the dataset to train the model.

- At the end of the log file, the model's accuracy is calculated at intervals on the evaluation dataset of HellaSwag, showing consistent improvement.

- It can be inferred that the model could have been trained to achieve accuracy comparable to the GPT-2 124M model if sufficient resources were available. Unfortunately, due to resource limitations, I was unable to complete the model's training.

## References

- Implementation inspired by Andrey Karpathy's work.
- Dataset: Fineweb (10B tokens sampled).
- Evaluation Dataset: HellaSwag.

---

For more details, check the associated training scripts and the uploaded resources in this repository.
