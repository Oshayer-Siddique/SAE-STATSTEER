import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = input("Model Name: ")

os.makedirs(f"outputs/{model_name}/multilayer", exist_ok=True)

with open("token.txt", "r") as f:
    token = f.read().strip()
with open("metadata.json", "r") as f:
    metadata = json.load(f)
with open("multilayer_metadata.json", "r") as f:
    ml_metadata = json.load(f)

model_id = f"google/{model_name}"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
    token=token,
)

DOMAINS = ['LOGIC', 'SENTIMENT', 'POLITIC', 'MORAL']


def get_layers(model):
    m = model.model
    if hasattr(m, 'language_model'):
        return m.language_model.layers
    if hasattr(m, 'layers'):
        return m.layers
    raise AttributeError(f"Cannot find layers in {type(m)}")


def ckpt_path(domain):
    return f"outputs/{model_name}/multilayer/{domain}_multilayer_checkpoint.json"


def load_checkpoint(domain):
    path = ckpt_path(domain)
    if os.path.exists(path):
        with open(path) as f:
            ckpt = json.load(f)
        completed = set(ckpt.get("completed_prompts", []))
        data      = ckpt.get("data", {"baseline": [], "steered": []})
        print(f"  [ckpt] Resuming {domain} multilayer — "
              f"{len(completed)} prompts already done")
        return data, completed
    return {"baseline": [], "steered": []}, set()


def save_checkpoint(domain, data, completed_prompts):
    with open(ckpt_path(domain), "w") as f:
        json.dump({
            "data":              data,
            "completed_prompts": list(completed_prompts),
        }, f, indent=2)


def create_steering_hook(alpha, steering_unit):
    def hook(module, input, output):
        if isinstance(output, tuple):
            h      = output[0]
            h_norm = h[:, -1, :].norm(dim=-1, keepdim=True)
            h      = h.clone()
            h[:, -1, :] = h[:, -1, :] + alpha * h_norm * steering_unit
            return (h,) + output[1:]
        return output
    return hook


def load_steering_unit(domain, layer):
    layer_str  = str(layer)
    vector_dir = metadata[model_name][domain][layer_str]
    input_file = f"vectors/{model_name}/{domain}/{vector_dir}/delta_h_init.npy"
    if not os.path.exists(input_file):
        input_file = f"vectors/{model_name}/{domain}/{vector_dir}/delta_h.npy"
    delta_h_np   = np.load(input_file)
    delta_h_unit = delta_h_np / (np.linalg.norm(delta_h_np) + 1e-8)
    return torch.from_numpy(delta_h_unit).to(model.device, dtype=torch.bfloat16)


for domain in DOMAINS:

    domain_config      = ml_metadata[model_name][domain]
    layers             = domain_config["layers"]
    weights            = domain_config["weights"]
    best_single_alphas = domain_config["alphas"]

    # ── Compute fixed per-layer alphas ───────────────────────────────────
    # α_layer = weight × best_single_alpha
    # Weight prevents compounding; best_single_alpha respects each layer's optimum
    per_layer_alphas = {}
    for layer, weight, best_alpha in zip(layers, weights, best_single_alphas):
        per_layer_alphas[layer] = round(weight * best_alpha, 6)

    print(f"\n{'='*70}")
    print(f"Model: {model_name} | Domain: {domain} | MULTI-LAYER (fixed α)")
    print(f"{'='*70}")
    print(f"  Per-layer config:")
    for l, w, ba in zip(layers, weights, best_single_alphas):
        print(f"    Layer {l:>2d}: weight={w:.3f} × α*={ba:.1f} → α={per_layer_alphas[l]:.4f}")

    # ── Pre-load all steering units ──────────────────────────────────────
    steering_units = {}
    for layer in layers:
        steering_units[layer] = load_steering_unit(domain, layer)
        print(f"  Loaded steering unit for layer {layer}")

    # ── Load prompts ─────────────────────────────────────────────────────
    with open(f"prompts/{domain.lower()}-prompts.txt") as f:
        prompts = [line.strip() for line in f if line.strip()]

    DATA, completed = load_checkpoint(domain)

    md_path     = f"outputs/{model_name}/multilayer/{domain}_multilayer_results.md"
    OUTPUT_FILE = open(md_path, "a")

    if not completed:
        OUTPUT_FILE.write(f"# Multi-Layer Steering: {domain}\n\n")
        OUTPUT_FILE.write(f"**Model:** {model_name}\n\n")
        OUTPUT_FILE.write(f"**Per-layer config:**\n\n")
        for l, w, ba in zip(layers, weights, best_single_alphas):
            OUTPUT_FILE.write(
                f"- Layer {l}: weight={w}, best_single_α={ba}, "
                f"applied_α={per_layer_alphas[l]:.4f}\n"
            )
        OUTPUT_FILE.write(f"\n---\n\n")
        OUTPUT_FILE.flush()

    # ── Main generation loop ─────────────────────────────────────────────
    for prompt in prompts:
        if prompt in completed:
            print(f"  [skip] {prompt[:60]}...")
            continue

        print(f"  Processing: {prompt[:70]}...")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        OUTPUT_FILE.write(f"# {prompt}\n\n")

        # ── Baseline (no hooks) ──────────────────────────────────────
        print(f"    [baseline]")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.2,
                pad_token_id=tokenizer.eos_token_id,
            )
        baseline_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        DATA["baseline"].append({
            "original_input": prompt,
            "generated":      baseline_text,
        })
        OUTPUT_FILE.write(f"## Baseline\n\n{baseline_text.strip()}\n\n")
        OUTPUT_FILE.flush()

        # ── Multi-layer steered ──────────────────────────────────────
        handles = []
        try:
            for layer in layers:
                handle = get_layers(model)[int(layer)].register_forward_hook(
                    create_steering_hook(per_layer_alphas[layer],
                                        steering_units[layer])
                )
                handles.append(handle)

            alpha_display = ", ".join(
                f"L{l}={per_layer_alphas[l]:.4f}" for l in layers
            )
            print(f"    [steered] {alpha_display}")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.2,
                    pad_token_id=tokenizer.eos_token_id,
                )
            steered_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            DATA["steered"].append({
                "original_input":   prompt,
                "generated":        steered_text,
                "per_layer_alphas": {str(l): per_layer_alphas[l] for l in layers},
            })

            OUTPUT_FILE.write(
                f"## Multi-Layer Steered [{alpha_display}]\n\n"
                f"{steered_text.strip()}\n\n"
            )
            OUTPUT_FILE.flush()

        finally:
            for handle in handles:
                handle.remove()

        completed.add(prompt)
        save_checkpoint(domain, DATA, completed)
        print(f"    [ckpt] Saved ({len(completed)}/{len(prompts)} prompts done)")

    OUTPUT_FILE.close()

    # ── Save final results ───────────────────────────────────────────
    json_path = f"outputs/{model_name}/multilayer/{domain}_multilayer_results.json"
    results_with_config = {
        "config": {
            "model":              model_name,
            "domain":             domain,
            "method":             "weighted_multilayer_fixed_alpha",
            "layers":             layers,
            "weights":            weights,
            "best_single_alphas": best_single_alphas,
            "applied_alphas":     {str(l): per_layer_alphas[l] for l in layers},
            "formula":            "alpha_layer = weight × best_single_alpha",
        },
        "results": DATA,
    }
    with open(json_path, "w") as f:
        json.dump(results_with_config, f, indent=4)

    print(f"  Done. Results saved to {json_path}")

print(f"\n{'='*70}")
print(f"ALL DOMAINS COMPLETE for {model_name}")
print(f"{'='*70}")