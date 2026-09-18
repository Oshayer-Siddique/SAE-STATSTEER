# STAGE 1: SAE Feature Extraction

> **Location:** [`STAGE-1-Features_extraction`](file:///E:/SAE-STATSTEER/STAGE-1-Features_extraction)  
> **Pipeline Phase:** Stage 1 of 4

---

## 🎯 1. Objective of this Folder

The objective of **Stage 1** is to extract high-dimensional Sparse Autoencoder (SAE) feature activation matrices from post-MLP residual streams of LLMs (such as Gemma 2 2B, Gemma 2 9B, and Gemma 3 4B).

Using topic-matched contrastive prompt pairs ($n_+ = 800$ target vs. $n_- = 800$ anti-target examples across domains like LOGIC, MORAL, POLITICS, and SENTIMENT), this stage:
1. Passes contrastive pairs through the frozen LLM.
2. Encodes residual stream activations into JumpReLU SAE feature activation vectors ($D_{\text{SAE}} = 16,384$).
3. Performs token-wise max-pooling across sequence length to capture localized feature evidence.
4. Applies a non-linear log compression $\tilde{z} = \log(1 + z)$ (`log1p`) to manage heavy-tailed feature distributions.

---

## 🚀 2. How to Run the Scripts & Execution Order

Execution must follow a 2-step sequence:

### Step 1: Pre-Flight Environment Check
Run `check_setup.py` to verify system dependencies, JAX TPU accessibility, Penzai model loader framework, and tokenizer availability.

```bash
python check_setup.py
```

### Step 2: Extract SAE Feature Activation Matrices
Run `01_extract_all_sae_features.py` to process contrast pairs and export layer-wise `.npz` feature activation matrices.

```bash
python SCRIPTS/01_extract_all_sae_features.py
```

---

## 📥 3. Required Inputs for Each Script

### Script 1: [`check_setup.py`](file:///E:/SAE-STATSTEER/STAGE-1-Features_extraction/check_setup.py)
* **Inputs:** None (system-level check).
* **Environment Requirements:**
  * Python $\ge 3.10$
  * `numpy`, `jax`, `jaxlib`, `flax`, `optax`, `penzai`, `orbax.checkpoint`, `transformers`, `huggingface_hub`
  * Active TPU backend (or fallback GPU/CPU)
  * Internet access or cached access to `google/gemma-2-2b` tokenizer from HuggingFace.

### Script 2: [`SCRIPTS/01_extract_all_sae_features.py`](file:///E:/SAE-STATSTEER/STAGE-1-Features_extraction/SCRIPTS/01_extract_all_sae_features.py)
* **Configuration Parameters (configured inside `CONFIG` dict):**
  * `model_path`: Directory path to pre-trained model Flax/Orbax checkpoint (e.g., Gemma 2 9B / Gemma 3 4B).
  * `vocab_path`: Path to `tokenizer.model` SentencePiece file.
  * `dataset_path`: JSON file containing 1,600 contrastive prompt pairs (e.g., `political_polarity.json` or domain-specific dataset) with `positive_text` / `negative_text` or target/anti-target labels.
  * `sae_base_dir`: Directory containing JumpReLU SAE dictionary weights per layer.
  * `target_layers`: List of integer target layers to probe (e.g., `[12, 19, 26, 31, 38]`).
  * `max_seq_len`: Token sequence length limit (default: `256`).

* **Output Artifacts Produced:**
  * `layer_XX_all_sae_features.npz` matrices of shape $(2N, 16384)$ for each target layer $XX$.
  * Periodic checkpoint files saved every 50 prompt pairs.
