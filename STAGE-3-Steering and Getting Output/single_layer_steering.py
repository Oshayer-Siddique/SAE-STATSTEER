import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = input("Model Name: ")

os.makedirs(f"outputs/{model_name}", exist_ok=True)

with open("token.txt", "r") as f:
    token = f.read().strip()
with open("metadata.json", "r") as f:
    metadata = json.load(f)

model_id = f"google/{model_name}"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
    token=token,
)

ALPHA_VALUES = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
DOMAINS      = ['LOGIC', 'SENTIMENT', 'POLITIC', 'MORAL']


def get_layers(model):
    m = model.model
    if hasattr(m, 'language_model'):
        return m.language_model.layers  # Gemma 3 4B
    if hasattr(m, 'layers'):
        return m.layers                 # Gemma 2 2B / 9B
    raise AttributeError(f"Cannot find layers in {type(m)}")


def ckpt_path(domain, layer):
    return f"outputs/{model_name}/{domain}_{layer}_checkpoint.json"


def load_checkpoint(domain, layer):
    path = ckpt_path(domain, layer)
    if os.path.exists(path):
        with open(path) as f:
            ckpt = json.load(f)
        completed = set(ckpt.get("completed_prompts", []))
        data      = ckpt.get("data", {"baseline": []})
        print(f"  [ckpt] Resuming {domain} layer {layer} — "
              f"{len(completed)} prompts already done")
        return data, completed
    return {"baseline": []}, set()


def save_checkpoint(domain, layer, data, completed_prompts):
    with open(ckpt_path(domain, layer), "w") as f:
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


for domain in DOMAINS:
    for target_layer in metadata[model_name][domain]["layers"]:

        print(f"\nModel: {model_name} | Domain: {domain} | Layer: {target_layer}")

        input_file = f"vectors/{model_name}/{domain}/{metadata[model_name][domain][target_layer]}/delta_h_init.npy"
        if not os.path.exists(input_file):
            input_file = f"vectors/{model_name}/{domain}/{metadata[model_name][domain][target_layer]}/delta_h.npy"

        delta_h_np    = np.load(input_file)
        delta_h_unit  = delta_h_np / (np.linalg.norm(delta_h_np) + 1e-8)
        steering_unit = torch.from_numpy(delta_h_unit).to(
            model.device, dtype=torch.bfloat16)

        with open(f"prompts/{domain.lower()}-prompts.txt") as f:
            prompts = [line.strip() for line in f if line.strip()]

        DATA, completed = load_checkpoint(domain, target_layer)

        md_path     = f"outputs/{model_name}/{domain}_{target_layer}_results.md"
        OUTPUT_FILE = open(md_path, "a")

        for prompt in prompts:
            if prompt in completed:
                print(f"  [skip] {prompt[:60]}...")
                continue

            print(f"  Processing: {prompt[:70]}...")
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            OUTPUT_FILE.write(f"# {prompt}\n")

            for alpha in ALPHA_VALUES:
                print(f"    alpha={alpha}")

                handle = None
                if alpha != 0:
                    handle = get_layers(model)[int(target_layer)].register_forward_hook(
                        create_steering_hook(alpha, steering_unit))
                try:
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=200,
                            do_sample=True,
                            temperature=0.2,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    generated_text = tokenizer.decode(
                        outputs[0], skip_special_tokens=True)

                    key = "baseline" if alpha == 0 else str(alpha)
                    if key not in DATA:
                        DATA[key] = []
                    DATA[key].append({
                        "original_input": prompt,
                        "generated":      generated_text,
                    })

                    OUTPUT_FILE.write(f"## Alpha {alpha}:\n\n{generated_text.strip()}\n\n")
                    OUTPUT_FILE.flush()

                finally:
                    if handle:
                        handle.remove()

            completed.add(prompt)
            save_checkpoint(domain, target_layer, DATA, completed)
            print(f"    [ckpt] Saved ({len(completed)}/{len(prompts)} prompts done)")

        OUTPUT_FILE.close()

        json_path = f"outputs/{model_name}/{domain}_{target_layer}_results.json"
        with open(json_path, "w") as f:
            json.dump(DATA, f, indent=4)

#        os.remove(ckpt_path(domain, target_layer))
        print(f"  Done. Results saved to {json_path}")