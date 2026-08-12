"""
Pre-flight check for gradient x activation attribution setup.
Verifies all dependencies, files, and basic functionality are in place.
"""

import sys
import os
import traceback

passed = []
failed = []
warnings = []

def check(name, fn):
    try:
        result = fn()
        if result is None or result is True:
            passed.append(name)
            print(f"  [PASS] {name}")
        else:
            passed.append(f"{name} ({result})")
            print(f"  [PASS] {name}: {result}")
    except Exception as e:
        failed.append((name, str(e)))
        print(f"  [FAIL] {name}: {e}")

def warn(name, msg):
    warnings.append((name, msg))
    print(f"  [WARN] {name}: {msg}")

print("=" * 65)
print("ATTRIBUTION SETUP CHECK")
print("=" * 65)

# --- 1. Python version ---
print("\n[1] Python version")
check("Python >= 3.10", lambda: (
    f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 10) else (_ for _ in ()).throw(RuntimeError("need >= 3.10"))
))

# --- 2. Core numerical stack ---
print("\n[2] Core numerical stack")
check("numpy", lambda: __import__("numpy").__version__)
check("jax", lambda: __import__("jax").__version__)
check("jaxlib", lambda: __import__("jaxlib").__version__)

# --- 3. JAX sees TPU ---
print("\n[3] TPU accessibility")
def check_tpu():
    import jax
    devices = jax.devices()
    tpu_count = sum(1 for d in devices if "Tpu" in type(d).__name__ or "tpu" in str(d).lower())
    if tpu_count == 0:
        raise RuntimeError(f"no TPU devices found (got {devices})")
    return f"{tpu_count} TPU chips visible"
check("JAX TPU backend", check_tpu)

def tpu_math():
    import jax
    import jax.numpy as jnp
    x = jnp.ones((100, 100))
    y = jnp.dot(x, x)
    s = float(y.sum())
    if abs(s - 1_000_000) > 1:
        raise RuntimeError(f"matmul produced wrong sum: {s}")
    return "matmul OK"
check("TPU matmul test", tpu_math)

# --- 4. Model framework ---
print("\n[4] Model framework")
check("flax", lambda: __import__("flax").__version__)
check("orbax.checkpoint", lambda: __import__("orbax.checkpoint").checkpoint.__name__)
check("optax", lambda: __import__("optax").__version__)
check("penzai", lambda: __import__("penzai").__version__)

def check_penzai_gemma():
    from penzai.models.transformer.variants.gemma import gemma_from_pretrained_checkpoint
    return "gemma_from_pretrained_checkpoint importable"
check("penzai Gemma loader", check_penzai_gemma)

def check_penzai_pz():
    from penzai import pz
    # named-array wrap should be available
    if not hasattr(pz, "nx"):
        raise RuntimeError("pz.nx missing")
    return "pz + pz.nx available"
check("penzai named arrays", check_penzai_pz)

# --- 5. Tokenizer stack ---
print("\n[5] Tokenizer stack")
check("transformers", lambda: __import__("transformers").__version__)
check("huggingface_hub", lambda: __import__("huggingface_hub").__version__)

def check_tokenizer():
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    # check the tokens we actually need
    results = {}
    for text in [" yes", " no", " A", " B", " C", " D"]:
        ids = t.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"{text!r} tokenizes to {len(ids)} tokens, expected 1")
        results[text] = ids[0]
    expected = {" yes": 7778, " no": 793, " A": 586, " B": 599}
    for k, v in expected.items():
        if results[k] != v:
            raise RuntimeError(f"{k!r} token ID mismatch: expected {v}, got {results[k]}")
    return f"all answer tokens single-token (yes={results[' yes']}, no={results[' no']})"
check("Gemma tokenizer + answer tokens", check_tokenizer)

# --- 6. Model weights on disk ---
print("\n[6] Model weights")
CHECKPOINT_DIR = "/home/oshayer_siddique2001/gemma-2-2b-flax/gemma2-2b"
TOKENIZER_PATH = "/home/oshayer_siddique2001/gemma-2-2b-flax/tokenizer.model"

def check_ckpt_dir():
    if not os.path.isdir(CHECKPOINT_DIR):
        raise RuntimeError(f"{CHECKPOINT_DIR} does not exist")
    required = ["_METADATA", "manifest.ocdbt", "checkpoint"]
    missing = [f for f in required if not os.path.exists(os.path.join(CHECKPOINT_DIR, f))]
    if missing:
        raise RuntimeError(f"missing files: {missing}")
    total_mb = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, fs in os.walk(CHECKPOINT_DIR) for f in fs
    ) / (1024 * 1024)
    if total_mb < 3000:
        raise RuntimeError(f"checkpoint only {total_mb:.0f} MB, expected >3000 MB")
    return f"{total_mb:.0f} MB total"
check("Checkpoint directory", check_ckpt_dir)

check("Tokenizer.model file", lambda: (
    f"{os.path.getsize(TOKENIZER_PATH) / 1024 / 1024:.1f} MB"
    if os.path.exists(TOKENIZER_PATH)
    else (_ for _ in ()).throw(RuntimeError("not found"))
))

# --- 7. Disk space ---
print("\n[7] Disk space")
def check_disk():
    import shutil
    total, used, free = shutil.disk_usage(os.path.expanduser("~"))
    free_gb = free / (1024 ** 3)
    if free_gb < 5:
        raise RuntimeError(f"only {free_gb:.1f} GB free, need at least 5 GB")
    return f"{free_gb:.1f} GB free"
check("Home directory free space", check_disk)

# --- 8. jax.grad works (the actual thing attribution needs) ---
print("\n[8] Autodiff test (core requirement for gradient x activation)")
def check_grad():
    import jax
    import jax.numpy as jnp
    def f(x):
        return jnp.sum(x ** 2)
    g = jax.grad(f)(jnp.array([1.0, 2.0, 3.0]))
    expected = jnp.array([2.0, 4.0, 6.0])
    if not jnp.allclose(g, expected):
        raise RuntimeError(f"grad wrong: {g} vs {expected}")
    return "jax.grad works"
check("jax.grad basic test", check_grad)

def check_grad_on_tpu():
    import jax
    import jax.numpy as jnp
    @jax.jit
    def loss(x):
        return jnp.sum(jax.nn.relu(x) ** 2)
    grad_fn = jax.jit(jax.grad(loss))
    x = jnp.array([1.0, -2.0, 3.0, -4.0])
    g = grad_fn(x)
    # expected: grad of sum(relu(x)^2) = 2*relu(x) for positive, 0 for negative
    expected = jnp.array([2.0, 0.0, 6.0, 0.0])
    if not jnp.allclose(g, expected):
        raise RuntimeError(f"jit grad wrong: {g} vs {expected}")
    return "JIT-compiled grad works on TPU"
check("JIT + grad on TPU", check_grad_on_tpu)

# --- Summary ---
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  Passed:   {len(passed)}")
print(f"  Failed:   {len(failed)}")
print(f"  Warnings: {len(warnings)}")

if failed:
    print("\n  FAILED CHECKS:")
    for name, err in failed:
        print(f"    - {name}: {err}")
    print("\n  ==> Fix the above before running attribution.")
    sys.exit(1)
elif warnings:
    print("\n  WARNINGS (non-blocking):")
    for name, msg in warnings:
        print(f"    - {name}: {msg}")
    print("\n  ==> Everything works, but review warnings.")
    sys.exit(0)
else:
    print("\n  ==> All checks passed. Ready for gradient x activation.")
    sys.exit(0)