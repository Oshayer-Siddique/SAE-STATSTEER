"""
Extract SAE features for ALL political_polarity pairs.

positive_text -> label 1 = right/conservative direction
negative_text -> label 0 = left/liberal direction

For N pairs:
    Z shape per layer = (2N, 16384)

Outputs:
    /home/oshayer/NEW_CONCEPT/political_polarity/results/sae_features/layer_XX_all_sae_features.npz

Checkpoints:
    saved every 50 pairs
"""

import os
import glob
import json
import time

import numpy as np
import jax
import jax.numpy as jnp

from penzai import pz
from penzai.models.transformer.variants.gemma import gemma_from_pretrained_checkpoint
import orbax.checkpoint as ocp
import sentencepiece as spm


print(f"JAX devices: {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")


CONFIG = {
    "model_path": "/home/oshayer/gemma-2-9b-flax/gemma2_9b_pt",
    "vocab_path": "/home/oshayer/gemma-2-9b-flax/tokenizer.model",
    "dataset_path": "/home/oshayer/NEW_CONCEPT/political_polarity/dataset/political_polarity.json",

    "sae_base_dir": "/home/oshayer/sae-9b/weights/gemma-scope-9b-pt-res",

    "target_layers": [12,19, 26,31,38],

    "max_seq_len": 256,

    "output_dir": "/home/oshayer/NEW_CONCEPT/political_polarity/results/sae_features",
    "checkpoint_dir": "/home/oshayer/NEW_CONCEPT/political_polarity/results/sae_features/checkpoints",

    "checkpoint_every": 50,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

TARGET_LAYERS = CONFIG["target_layers"]


# ============================================================
# DATASET
# ============================================================

def load_dataset(path):
    with open(path, "r") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            data = json.load(f)
        else:
            data = []
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

    print(f"Loaded {len(data)} pairs from {path}")
    return data


def get_split(sample):
    return str(sample.get("split", "unknown")).strip().lower()


def get_split2(sample):
    return str(sample.get("split2", "unknown")).strip().lower()


def get_meta(sample, key, default="unknown"):
    return str(sample.get("meta", {}).get(key, default)).strip().lower()


# ============================================================
# TOKENIZER
# ============================================================

def load_tokenizer():
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(CONFIG["vocab_path"])
    print(f"Tokenizer loaded. Vocab size: {tokenizer.GetPieceSize()}")
    return tokenizer


def tokenize_text(text, tokenizer, max_len):
    token_ids = tokenizer.EncodeAsIds(text)
    token_ids = [tokenizer.bos_id()] + token_ids

    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]

    return np.array(token_ids, dtype=np.int32)


# ============================================================
# GEMMA MODEL
# ============================================================

def load_gemma_model():
    print("Loading Gemma 2 9B via Penzai...")
    print("Checkpoint:", CONFIG["model_path"])

    ckpt = ocp.PyTreeCheckpointer().restore(CONFIG["model_path"])

    model = gemma_from_pretrained_checkpoint(
        ckpt,
        preset_name="gemma2_9b",
        upcast_activations_to_float32=True,
    )

    print(f"Model loaded successfully. Type: {type(model).__name__}")
    return model


def forward_pass_with_hidden_states(token_ids_array, model, target_layers):
    tokens_named = pz.nx.wrap(token_ids_array).tag("seq")

    seq_len = token_ids_array.shape[0]
    positions = pz.nx.wrap(jnp.arange(seq_len, dtype=jnp.int32)).tag("seq")

    side_inputs = {"token_positions": positions}

    PRE_OFFSET = 3
    NUM_TRANSFORMER_BLOCKS = 42

    captured = {}
    x = tokens_named

    for i in range(PRE_OFFSET):
        x = model.body.sublayers[i](x, **side_inputs)

    for layer_idx in range(NUM_TRANSFORMER_BLOCKS):
        sublayer_idx = PRE_OFFSET + layer_idx
        x = model.body.sublayers[sublayer_idx](x, **side_inputs)

        if layer_idx in target_layers:
            captured[layer_idx] = x.unwrap("seq", "embedding")

    for i in range(PRE_OFFSET + NUM_TRANSFORMER_BLOCKS, len(model.body.sublayers)):
        x = model.body.sublayers[i](x, **side_inputs)

    logits = x.unwrap("seq", "vocabulary")
    return logits, captured


def extract_last_token_hiddens(text, tokenizer, model, target_layers):
    token_ids = tokenize_text(text, tokenizer, CONFIG["max_seq_len"])
    token_ids_jax = jnp.array(token_ids)

    _, hidden_states = forward_pass_with_hidden_states(
        token_ids_jax,
        model,
        target_layers,
    )

    out = {}
    for layer in target_layers:
        h_last = hidden_states[layer][-1, :]
        out[layer] = np.array(h_last, dtype=np.float32)

    return out


# ============================================================
# SAE
# ============================================================

def find_sae_params_path(layer):
    pattern = os.path.join(
        CONFIG["sae_base_dir"],
        f"layer_{layer}",
        "width_16k",
        "*",
        "params.npz",
    )

    matches = sorted(glob.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No SAE params found for layer {layer}. Pattern searched:\n{pattern}"
        )

    if len(matches) > 1:
        print(f"WARNING: multiple SAE params found for layer {layer}. Using first:")
        for m in matches:
            print(" ", m)

    return matches[0]


def pick_key(keys, candidates):
    for c in candidates:
        if c in keys:
            return c
    return None


class GemmaScopeSAE:
    def __init__(self, params_path):
        self.params_path = params_path

        data = np.load(params_path)
        keys = list(data.keys())

        print("\nLoading SAE:")
        print(" ", params_path)
        print("Available keys:", keys)

        w_enc_key = pick_key(keys, ["W_enc", "w_enc", "encoder.W", "encoder_w"])
        b_enc_key = pick_key(keys, ["b_enc", "encoder.b", "encoder_b"])
        b_dec_key = pick_key(keys, ["b_dec", "decoder.b", "decoder_b"])
        threshold_key = pick_key(keys, ["threshold", "thresholds", "theta", "jump_threshold"])

        if w_enc_key is None:
            raise KeyError(f"Could not find encoder weight key. Keys={keys}")
        if b_enc_key is None:
            raise KeyError(f"Could not find encoder bias key. Keys={keys}")
        if b_dec_key is None:
            raise KeyError(f"Could not find decoder bias key. Keys={keys}")

        self.W_enc = np.array(data[w_enc_key], dtype=np.float32)
        self.b_enc = np.array(data[b_enc_key], dtype=np.float32)
        self.b_dec = np.array(data[b_dec_key], dtype=np.float32)

        if threshold_key is not None:
            self.threshold = np.array(data[threshold_key], dtype=np.float32)
            print(f"Using JumpReLU threshold key: {threshold_key}")
        else:
            self.threshold = None
            print("No threshold found. Using ReLU encoder.")

        self.hidden_dim = int(self.b_dec.shape[0])
        self.sae_dim = int(self.b_enc.shape[0])

        print(f"W_enc shape: {self.W_enc.shape}")
        print(f"b_enc shape: {self.b_enc.shape}")
        print(f"b_dec shape: {self.b_dec.shape}")
        if self.threshold is not None:
            print(f"threshold shape: {self.threshold.shape}")

        self.W_enc_jax = jnp.array(self.W_enc)
        self.b_enc_jax = jnp.array(self.b_enc)
        self.b_dec_jax = jnp.array(self.b_dec)

        if self.threshold is not None:
            self.threshold_jax = jnp.array(self.threshold)
        else:
            self.threshold_jax = None

        self._encode_batch_jit = jax.jit(self._encode_batch)

    def _encode_batch(self, h_batch):
        h_centered = h_batch - self.b_dec_jax

        if self.W_enc.shape[0] == self.hidden_dim:
            pre = h_centered @ self.W_enc_jax + self.b_enc_jax
        elif self.W_enc.shape[1] == self.hidden_dim:
            pre = h_centered @ self.W_enc_jax.T + self.b_enc_jax
        else:
            raise ValueError(
                f"W_enc shape {self.W_enc.shape} incompatible with hidden_dim={self.hidden_dim}"
            )

        if self.threshold_jax is not None:
            z = jnp.where(pre > self.threshold_jax, pre, 0.0)
        else:
            z = jnp.maximum(pre, 0.0)

        return z

    def encode_batch(self, h_batch_np):
        h_batch = jnp.array(h_batch_np, dtype=jnp.float32)

        if h_batch.shape[1] != self.hidden_dim:
            raise ValueError(
                f"Hidden dim mismatch. Got {h_batch.shape[1]}, SAE expects {self.hidden_dim}"
            )

        z = self._encode_batch_jit(h_batch)
        return np.array(z, dtype=np.float32)


def load_saes_for_layers(layers):
    saes = {}
    for layer in layers:
        params_path = find_sae_params_path(layer)
        saes[layer] = GemmaScopeSAE(params_path)
    return saes


# ============================================================
# CHECKPOINTING
# ============================================================

def latest_checkpoint_path():
    return os.path.join(CONFIG["checkpoint_dir"], "checkpoint_latest_all_layers.npz")


def numbered_checkpoint_path(pair_count):
    return os.path.join(
        CONFIG["checkpoint_dir"],
        f"checkpoint_{pair_count:04d}_all_layers.npz",
    )


def save_checkpoint(pair_count, layer_Z, meta_rows):
    save_dict = {
        "processed_pair_count": np.array(pair_count, dtype=np.int32),
        "pair_ids": np.array([r["pair_id"] for r in meta_rows]),
        "sides": np.array([r["side"] for r in meta_rows]),
        "labels": np.array([r["label"] for r in meta_rows], dtype=np.int32),
        "split": np.array([r["split"] for r in meta_rows]),
        "split2": np.array([r["split2"] for r in meta_rows]),
        "topic": np.array([r["topic"] for r in meta_rows]),
        "category": np.array([r["category"] for r in meta_rows]),
        "target": np.array([r["target"] for r in meta_rows]),
        "sources": np.array([r["source"] for r in meta_rows]),
    }

    for layer in TARGET_LAYERS:
        save_dict[f"Z_layer_{layer}"] = np.stack(layer_Z[layer], axis=0).astype(np.float32)

    latest_path = latest_checkpoint_path()
    numbered_path = numbered_checkpoint_path(pair_count)

    np.savez(latest_path, **save_dict)
    np.savez(numbered_path, **save_dict)

    print("\nCheckpoint saved:")
    print(f"  latest:   {latest_path}")
    print(f"  numbered: {numbered_path}")


def load_checkpoint_if_available():
    path = latest_checkpoint_path()

    if not os.path.exists(path):
        print("\nNo checkpoint found. Starting from 0.")
        layer_Z = {layer: [] for layer in TARGET_LAYERS}
        meta_rows = []
        return 0, layer_Z, meta_rows

    print(f"\nLoading checkpoint: {path}")
    data = np.load(path, allow_pickle=True)

    processed_pair_count = int(data["processed_pair_count"])

    pair_ids = data["pair_ids"]
    sides = data["sides"]
    labels = data["labels"]
    split = data["split"]
    split2 = data["split2"]
    topic = data["topic"]
    category = data["category"]
    target = data["target"]
    sources = data["sources"]

    meta_rows = []
    for i in range(len(pair_ids)):
        meta_rows.append(
            {
                "pair_id": str(pair_ids[i]),
                "side": str(sides[i]),
                "label": int(labels[i]),
                "split": str(split[i]),
                "split2": str(split2[i]),
                "topic": str(topic[i]),
                "category": str(category[i]),
                "target": str(target[i]),
                "source": str(sources[i]),
            }
        )

    layer_Z = {}
    for layer in TARGET_LAYERS:
        Z = data[f"Z_layer_{layer}"]
        layer_Z[layer] = [Z[i].astype(np.float32) for i in range(Z.shape[0])]

    print(f"Resuming from pair index: {processed_pair_count}")
    print(f"Loaded rows per layer: {len(layer_Z[TARGET_LAYERS[0]])}")

    return processed_pair_count, layer_Z, meta_rows


# ============================================================
# FINAL SAVE
# ============================================================

def save_final_outputs(layer_Z, meta_rows):
    pair_ids = np.array([r["pair_id"] for r in meta_rows])
    sides = np.array([r["side"] for r in meta_rows])
    y = np.array([r["label"] for r in meta_rows], dtype=np.int32)
    split = np.array([r["split"] for r in meta_rows])
    split2 = np.array([r["split2"] for r in meta_rows])
    topic = np.array([r["topic"] for r in meta_rows])
    category = np.array([r["category"] for r in meta_rows])
    target = np.array([r["target"] for r in meta_rows])
    sources = np.array([r["source"] for r in meta_rows])

    for layer in TARGET_LAYERS:
        Z = np.stack(layer_Z[layer], axis=0).astype(np.float32)

        z_right = Z[y == 1]
        z_left = Z[y == 0]

        output_path = os.path.join(
            CONFIG["output_dir"],
            f"layer_{layer}_all_sae_features.npz",
        )

        np.savez(
            output_path,
            Z=Z,
            y=y,
            z_right=z_right,
            z_left=z_left,
            pair_ids=pair_ids,
            sides=sides,
            split=split,
            split2=split2,
            topic=topic,
            category=category,
            target=target,
            sources=sources,
            layer=np.array(layer, dtype=np.int32),
            method=np.array("political_polarity_all_pairs_last_token_residual_to_sae_features"),
        )

        nonzero_per_row = np.count_nonzero(Z, axis=1)

        summary = {
            "output_path": output_path,
            "layer": int(layer),
            "Z_shape": list(Z.shape),
            "z_right_shape": list(z_right.shape),
            "z_left_shape": list(z_left.shape),
            "num_right": int(np.sum(y == 1)),
            "num_left": int(np.sum(y == 0)),
            "num_train_rows": int(np.sum(split == "train")),
            "num_probe_rows": int(np.sum(split2 == "probe")),
            "num_attribution_rows": int(np.sum(split2 == "attribution")),
            "num_heldout_rows": int(np.sum((split2 == "heldout") | (split2 == "held_out"))),
            "mean_l0": float(np.mean(nonzero_per_row)),
            "median_l0": float(np.median(nonzero_per_row)),
            "min_l0": int(np.min(nonzero_per_row)),
            "max_l0": int(np.max(nonzero_per_row)),
            "mean_activation": float(np.mean(Z)),
            "max_activation": float(np.max(Z)),
        }

        summary_path = output_path.replace(".npz", "_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("\nSaved final features:")
        print(f"  {output_path}")
        print(f"  {summary_path}")
        print(f"  Z shape: {Z.shape}")
        print(f"  Mean L0: {summary['mean_l0']:.2f}")


# ============================================================
# MAIN
# ============================================================

def main():
    data = load_dataset(CONFIG["dataset_path"])

    print("\nDataset counts:")
    split_counts = {}
    split2_counts = {}
    topic_counts = {}
    category_counts = {}

    for x in data:
        s = get_split(x)
        split_counts[s] = split_counts.get(s, 0) + 1

        s2 = get_split2(x)
        split2_counts[s2] = split2_counts.get(s2, 0) + 1

        topic = get_meta(x, "topic")
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

        category = get_meta(x, "category")
        category_counts[category] = category_counts.get(category, 0) + 1

    print("split:", split_counts)
    print("split2:", split2_counts)
    print("num topics:", len(topic_counts))
    print("num categories:", len(category_counts))

    tokenizer = load_tokenizer()
    model = load_gemma_model()
    saes = load_saes_for_layers(TARGET_LAYERS)

    start_pair_idx, layer_Z, meta_rows = load_checkpoint_if_available()

    total_pairs = len(data)
    start_time = time.time()

    for idx in range(start_pair_idx, total_pairs):
        sample = data[idx]

        pair_id = sample.get("id", f"sample_{idx}")
        split = get_split(sample)
        split2 = get_split2(sample)

        topic = get_meta(sample, "topic")
        category = get_meta(sample, "category")
        target = str(sample.get("target", "right_vs_left"))
        source = str(sample.get("source", "unknown"))

        if idx % 10 == 0:
            elapsed = time.time() - start_time
            done_now = idx - start_pair_idx
            rate = done_now / elapsed if elapsed > 0 else 0.0
            remaining = total_pairs - idx
            eta = remaining / rate if rate > 0 else 0.0

            print(
                f"\n[{idx}/{total_pairs}] {pair_id} "
                f"| processed this run: {done_now} "
                f"| {rate:.3f} pairs/s "
                f"| ETA {eta / 60:.1f} min"
            )

        positive_text = sample["positive_text"]
        negative_text = sample["negative_text"]

        # positive_text = right-leaning
        # negative_text = left-leaning
        pos_hiddens = extract_last_token_hiddens(
            positive_text,
            tokenizer,
            model,
            TARGET_LAYERS,
        )

        neg_hiddens = extract_last_token_hiddens(
            negative_text,
            tokenizer,
            model,
            TARGET_LAYERS,
        )

        for layer in TARGET_LAYERS:
            h_batch = np.stack(
                [pos_hiddens[layer], neg_hiddens[layer]],
                axis=0,
            ).astype(np.float32)

            z_batch = saes[layer].encode_batch(h_batch)

            layer_Z[layer].append(z_batch[0])
            layer_Z[layer].append(z_batch[1])

        meta_rows.append(
            {
                "pair_id": pair_id,
                "side": "right",
                "label": 1,
                "split": split,
                "split2": split2,
                "topic": topic,
                "category": category,
                "target": target,
                "source": source,
            }
        )

        meta_rows.append(
            {
                "pair_id": pair_id,
                "side": "left",
                "label": 0,
                "split": split,
                "split2": split2,
                "topic": topic,
                "category": category,
                "target": target,
                "source": source,
            }
        )

        processed_pair_count = idx + 1

        if processed_pair_count % CONFIG["checkpoint_every"] == 0:
            save_checkpoint(
                processed_pair_count,
                layer_Z,
                meta_rows,
            )

    save_final_outputs(layer_Z, meta_rows)

    save_checkpoint(
        total_pairs,
        layer_Z,
        meta_rows,
    )

    print("\n" + "=" * 80)
    print("DONE: extracted political polarity SAE features for all pairs.")
    print("=" * 80)
    print(f"Output folder: {CONFIG['output_dir']}")
    print(f"Checkpoint folder: {CONFIG['checkpoint_dir']}")


if __name__ == "__main__":
    main()
