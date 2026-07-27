from __future__ import annotations
# %% [markdown]
# NLA round-trip accuracy degradation experiment.
# 
# Idea: patch a single token's activation at a single layer with its
# NLA round-trip reconstruction 
# $$
# (h -> AV(h) -> text -> AR(text) -> h_hat)
# $$
# then let generation continue normally. Compare downstream task accuracy
# against (a) an unpatched baseline and (b) a no-op patch control that
# re-injects the *original* activation unchanged.
# 
# This script assumes you have:
#   - a HF-loadable Gemma model (`MODEL_NAME`)
#   - your own trained AV and AR modules from natural_language_autoencoders,
#     exposed as callables: av(h_vec) -> str, ar(text) -> torch.Tensor
#   - GSM8K test split via `datasets`
# 
# Fill in the two TODOs (load_av_ar, extract_final_answer) for your repo's
# actual API / answer-parsing format before running.

# %%
VISIBLE_GPUS = "0,1"

# %% [markdown]
# ## Experiment Overview
# 
# All experiments share one AV server and use the same target tokenizer/model state where possible. The notebook deliberately unloads the 27B target before loading the AR, then reloads the target for introspection, so the full target and AR do not coexist on the notebook GPU.
# 
# Results are printed and displayed near the cells that produce them. Round-trip and paraphrasing summaries are also written beneath the configurable experiment output root; introspection and zero-vector results remain notebook-only.

# %%


# -----------------------------------------------------------------------------
# EDIT THESE VALUES FOR YOUR CLUSTER ALLOCATION
# -----------------------------------------------------------------------------
# USER_NAME is your Linux username / /home folder name.
# VISIBLE_GPUS are the physical GPU IDs assigned to you, comma-separated.
# After CUDA masking, the first listed physical GPU becomes cuda:0 in this notebook.
# SGLANG_PHYSICAL_GPU is the physical GPU reserved for the SGLang AV server.
USER_NAME = "kaylee"
VISIBLE_GPUS = "0,1"
SGLANG_PHYSICAL_GPU = "1"
LOCAL_DEVICE = "cuda:0"
# -----------------------------------------------------------------------------

# Run this notebook from a fresh kernel. CUDA_VISIBLE_DEVICES must be set before
# importing torch, so this setup cell intentionally defines paths/GPUs first.
import os
from pathlib import Path

HOME = Path("/home") / USER_NAME
NLA_REPO = Path(os.environ.get("NLA_REPO", HOME / "natural_language_autoencoders"))
SGLANG_REPO = Path(os.environ.get("SGLANG_REPO", HOME / "sglang"))
VENV_PYTHON = Path(os.environ.get("NLA_VENV_PYTHON", NLA_REPO / ".venv/bin/python"))
HF_CACHE = Path(os.environ.get("HF_HOME", HOME / ".cache/huggingface"))
ROOT = Path(os.environ.get("NLA_EXPERIMENT_ROOT", HOME / "experiments/nla_experiments"))
LIBNUMA_DIR = Path(
    os.environ.get(
        "NLA_LIBNUMA_DIR", NLA_REPO / "vendor/libnuma/usr/lib/x86_64-linux-gnu"
    )
)
PYTHON_INCLUDE_DIRS = [
    Path(
        os.environ.get(
            "NLA_PYTHON_INCLUDE",
            NLA_REPO / "vendor/python3.10-dev/usr/include/python3.10",
        )
    ),
    Path(
        os.environ.get(
            "NLA_PYTHON_MULTIARCH_INCLUDE",
            NLA_REPO / "vendor/python3.10-dev/usr/include",
        )
    ),
]
ROOT.mkdir(parents=True, exist_ok=True)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = VISIBLE_GPUS
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
if LIBNUMA_DIR.exists():
    os.environ["LD_LIBRARY_PATH"] = (
        f"{LIBNUMA_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    )
_existing_cpath = os.environ.get("CPATH", "")
_include_paths = [str(path) for path in PYTHON_INCLUDE_DIRS if path.exists()]
if _include_paths:
    os.environ["CPATH"] = ":".join([*_include_paths, _existing_cpath])

import gc
import math
import random
import re
import subprocess
import time
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Local NLA inference helpers from the released repo.
import sys

sys.path.insert(0, str(NLA_REPO))
from nla_inference import NLAClient, NLACritic

TARGET_MODEL = "google/gemma-3-27b-it"
ACTOR_REPO_ID = os.environ.get("NLA_ACTOR_DIR", "kitft/nla-gemma3-27b-L41-av")
AR_REPO_ID = os.environ.get("NLA_AR_DIR", "kitft/nla-gemma3-27b-L41-ar")
SGLANG_URL = os.environ.get("SGLANG_URL", "http://127.0.0.1:30000")
SGLANG_LOG = Path.home() / "logs" / "sglang_av_server.log"
SGLANG_LOG.parent.mkdir(parents=True, exist_ok=True)
LAYER_INDEX = 41
DEVICE = LOCAL_DEVICE
SEED = 1234
N_EXAMPLES = 200


def resolve_checkpoint(path_or_repo_id: str) -> str:
    """Return a local checkpoint path with nla_meta.yaml available."""
    candidate = Path(path_or_repo_id).expanduser()
    if candidate.exists():
        assert (
            candidate / "nla_meta.yaml"
        ).exists(), f"Missing nla_meta.yaml in {candidate}"
        return str(candidate)
    local_path = Path(snapshot_download(repo_id=path_or_repo_id))
    assert (
        local_path / "nla_meta.yaml"
    ).exists(), f"Missing nla_meta.yaml in downloaded snapshot {local_path}"
    return str(local_path)


ACTOR_DIR = resolve_checkpoint(ACTOR_REPO_ID)
AR_DIR = resolve_checkpoint(AR_REPO_ID)

assert NLA_REPO.exists(), f"Missing NLA repo: {NLA_REPO}"
assert SGLANG_REPO.exists(), f"Missing patched SGLang checkout: {SGLANG_REPO}"
assert VENV_PYTHON.exists(), f"Missing NLA venv Python: {VENV_PYTHON}"

print("home:", HOME)
print("NLA repo:", NLA_REPO)
print("SGLang checkout:", SGLANG_REPO)
print("HF cache:", HF_CACHE)
print("experiment output:", ROOT)
print("vendored libnuma:", LIBNUMA_DIR if LIBNUMA_DIR.exists() else "not found")
print("vendored Python includes:", [p for p in PYTHON_INCLUDE_DIRS if p.exists()])
print("visible physical GPUs:", VISIBLE_GPUS)
print("SGLang physical GPU:", SGLANG_PHYSICAL_GPU)
print("notebook device:", DEVICE)
print("actor checkpoint:", ACTOR_DIR)
print("AR checkpoint:", AR_DIR)
print(torch.__version__, torch.cuda.is_available())
if torch.cuda.is_available():
    print("visible cuda devices:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"cuda:{i}", torch.cuda.get_device_name(i))

UV_PYTHON_INCLUDE = "/home/kaylee/.local/share/uv/python/cpython-3.10.20-linux-x86_64-gnu/include/python3.10"
assert os.path.exists(os.path.join(UV_PYTHON_INCLUDE, "Python.h")), "Python.h not found at that path"

os.environ["CPATH"] = f"{UV_PYTHON_INCLUDE}:{os.environ.get('CPATH', '')}"
print("CPATH now:", os.environ["CPATH"])

REAL_LIBNUMA_DIR = "/home/kaylee/.local/lib"
assert os.path.exists(os.path.join(REAL_LIBNUMA_DIR, "libnuma.so.1")), "libnuma.so.1 not found there"

os.environ["LD_LIBRARY_PATH"] = f"{REAL_LIBNUMA_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}"
print("LD_LIBRARY_PATH now:", os.environ["LD_LIBRARY_PATH"])

# !rm -rf ~/.triton/cache

print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
print("CPATH:", os.environ.get("CPATH"))

# %% [markdown]
# ## Start The AV Server
# 
# The AV uses SGLang because the checkpoint expects `input_embeds` injection. This notebook launches the patched editable SGLang checkout through the NLA `.venv`, pinned to the physical GPU chosen in the setup cell.
# 
# 

# %%
def launch_sglang_actor() -> subprocess.Popen:
    env = os.environ.copy()
    env["HF_HOME"] = str(HF_CACHE)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = SGLANG_PHYSICAL_GPU
    env["SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR"] = "1"
    env["PYTHONPATH"] = f"{NLA_REPO}:{env.get('PYTHONPATH', '')}"
    if LIBNUMA_DIR.exists():
        env["LD_LIBRARY_PATH"] = f"{LIBNUMA_DIR}:{env.get('LD_LIBRARY_PATH', '')}"
    include_paths = [str(path) for path in PYTHON_INCLUDE_DIRS if path.exists()]
    if include_paths:
        env["CPATH"] = ":".join([*include_paths, env.get("CPATH", "")])

    cmd = [
        str(VENV_PYTHON),
        "-m",
        "sglang.launch_server",
        "--model-path",
        ACTOR_DIR,
        "--port",
        "30000",
        "--host",
        "127.0.0.1",
        "--disable-radix-cache",
        "--mem-fraction-static",
        "0.80",
        "--context-length",
        "512",
        "--attention-backend",
        "triton",
        "--disable-cuda-graph",
        "--trust-remote-code",
        "--log-level",
        "warning",
    ]
    print("Launching SGLang on physical GPU", SGLANG_PHYSICAL_GPU)
    print(" ".join(cmd))
    print("SGLang server logs ->", SGLANG_LOG)
    log_f = open(SGLANG_LOG, "ab", buffering=0)
    return subprocess.Popen(
        cmd, env=env, cwd=str(NLA_REPO), stdout=log_f, stderr=subprocess.STDOUT
    )



def sglang_is_healthy() -> bool:
    import urllib.request

    try:
        urllib.request.urlopen(SGLANG_URL + "/health", timeout=2).read()
        return True
    except Exception:
        return False


def wait_for_sglang(timeout_s: int = 300) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        if sglang_is_healthy():
            print("SGLang is healthy")
            return
        try:
            import urllib.request

            urllib.request.urlopen(SGLANG_URL + "/health", timeout=2).read()
        except Exception as exc:
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"SGLang did not become healthy: {last_error!r}")


sglang_proc = None
if sglang_is_healthy():
    print("SGLang is already healthy at", SGLANG_URL)
else:
    sglang_proc = launch_sglang_actor()
    wait_for_sglang()

# %%
print(sglang_is_healthy())

# %% [markdown]
# # Experiment 1 - GSM8k (Deprecated)

# %%
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    TARGET_MODEL, torch_dtype=torch.bfloat16, device_map={"": DEVICE},
    trust_remote_code=True, attn_implementation="eager",
).eval()
del model.model.vision_tower
gc.collect()
torch.cuda.empty_cache()
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# %%
print("model" in globals())
if "model" in globals():
    print(model.device if hasattr(model, "device") else next(model.parameters()).device)

# %%
client = NLAClient(ACTOR_DIR, sglang_url=SGLANG_URL)
critic = NLACritic(AR_DIR, device="cpu", dtype=torch.float32)
print("Target and AV client ready")


# %% [markdown]
# The reconstruction error is
# $$
# \mathcal{L} = \mathbb{E}_{h_l \sim \mathcal{H}} \mathbb{E}_{z \sim AV(\cdot | h_l)} [||h_l - AR(z)||_2^2]
# $$
# where $\mathcal{H}$ is the distribution produced by extracting layer $l$ activations from $M$ on a corpus of text. 
# 
# The FVE is
# $$
# 1 - \frac{L}{ \mathbb{E}_{h_l \sim \mathcal{H}} ||h_l - \bar{h_l}||^2_2}
# $$

# %%
import inspect
print(inspect.getsourcefile(client.generate))
print(inspect.getsource(client.generate))

# %%
# N_SANITY = 200
# MAX_NEW_TOKENS = 256

def av(h: torch.Tensor) -> str:
    v = h.detach().float().cpu().numpy()
    return client.generate(v, temperature=0.0, max_new_tokens=384, extract_explanation=True)

# PARAPHRASE_PROMPT = (
#     "Paraphrase the following explanation, preserving its meaning exactly "
#     "but using different wording:\n\n{text}\n\nParaphrase:"
# )

PARAPHRASE_PROMPT = (
    "Please paraphrase the text provided below, preserving its exact meaning "
    "but expressing it in substantially different words and sentence structure "
    "than the original. Do not simply swap individual words for synonyms -- "
    "restructure the sentence(s) genuinely.\n\n"
    "<text>\n{text}\n</text>\n\n"
    "Include your final paraphrased text in <transformed_text> tags."
)

# def paraphrase(text: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
#     prompt = PARAPHRASE_PROMPT.format(text=text)
#     inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
#     with torch.no_grad():
#         out = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#         )
#     return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

import re 

def paraphrase(text: str, max_new_tokens: int = 512) -> str:
    prompt = PARAPHRASE_PROMPT.format(text=text)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    match = re.search(r"<transformed_text>(.*?)</transformed_text>", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        print(f"[paraphrase] WARNING: no <transformed_text> tags. Raw[:200]={raw[:200]!r}")
        return raw

def ar(text: str) -> torch.Tensor:
    return critic.reconstruct(text)  # raw, unnormalized -- matches h's scale


# --- Patch hook: replace the residual stream at the last prompt token ---
class LastTokenPatcher:
    def __init__(self, patch_vec=None):
        self.patch_vec = patch_vec
        self.captured = None
        self._done = False  # only patch once

    def __call__(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # Only act on the initial multi-token prefill pass, not each
        # single-token decode step that follows.
        if hidden.shape[1] > 1 and not self._done:
            self.captured = hidden[0, -1, :].detach().clone()
            if self.patch_vec is not None:
                hidden = hidden.clone()
                hidden[0, -1, :] = self.patch_vec.to(hidden.dtype).to(hidden.device)
            self._done = True
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
        return output


def mse_cos_from_reconstruction(h_hat: torch.Tensor, h_orig: torch.Tensor, mse_scale: float):
    """Per-example, norm-matched metrics (for reporting alongside FVE, not used *in* FVE)."""
    pred = h_hat.float()
    gold = h_orig.float().cpu()
    pred_n = pred / pred.norm().clamp_min(1e-12) * mse_scale
    gold_n = gold / gold.norm().clamp_min(1e-12) * mse_scale
    mse = ((pred_n - gold_n) ** 2).mean().item()
    cos = (pred_n @ gold_n / (pred_n.norm() * gold_n.norm())).item()
    return mse, cos


def raw_sq_error(h_hat: torch.Tensor, h_orig: torch.Tensor) -> float:
    """||h_orig - h_hat||^2 on raw, unnormalized activations -- this is what
    the paper's L and FVE are defined over, so it must NOT be norm-rescaled
    the way mse_cos_from_reconstruction's inputs are."""
    assert h_hat.dim() == 1 and h_orig.dim() == 1, f"{h_hat.shape=} {h_orig.shape=}"
    return ((h_orig.float().cpu() - h_hat.float().cpu()) ** 2).sum().item()


def fve(sq_errors: list, h_orig_list: list) -> float:
    """FVE = 1 - E[||h - AR(z)||^2] / E[||h - h_bar||^2], both expectations
    taken over the same sample. h_bar is estimated as the sample mean of
    h_orig_list -- this must be computed over the full run, not per-example."""
    h_stack = torch.stack([h.float().cpu() for h in h_orig_list])  # [N, d]
    h_bar = h_stack.mean(dim=0)
    variance = ((h_stack - h_bar) ** 2).sum(dim=1).mean().item()
    L = sum(sq_errors) / len(sq_errors)
    return 1 - L / variance


def get_layer_module(m, layer_idx):
    return m.model.language_model.layers[layer_idx]

# https://github.com/kitft/natural_language_autoencoders/blob/main/examples/gemma27b_layer41_step6000.txt
VAR_NRM = 0.0579  # Gemma-3-27B / layer-41 predict-the-mean MSE from training set

def fve_nrm(mse_nrm: float) -> float:
    """Fraction of training-set activation variance explained (normalized space)."""
    return 1.0 - float(mse_nrm) / VAR_NRM


def run_with_patch(prompt: str, patch_vec, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    patcher = LastTokenPatcher(patch_vec)
    handle = get_layer_module(model, LAYER_INDEX).register_forward_hook(patcher)
    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        handle.remove()
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text, patcher.captured


# # --- Run the three-condition comparison on a GSM8K subset ---
# import re
# from datasets import load_dataset

# def extract_final_answer(text: str):
#     m = re.search(r"[-+]?\d[\d,]*\.?\d*", text.strip().split("\n")[-1])
#     return m.group().replace(",", "") if m else None

# def gsm8k_gold(ex):
#     return ex["answer"].split("####")[-1].strip().replace(",", "")

# ds = load_dataset("gsm8k", "main", split="test").select(range(N_SANITY))
# correct = {"baseline": 0, "nla_roundtrip": 0, "paraphrase_roundtrip": 0}

# h_orig_list = []
# h_hat_list, h_hat_para_list = [], []
# cos_list, mse_list, sq_err_list = [], [], []
# cos_para_list, mse_para_list, sq_err_para_list = [], [], []
# mse_nrm_list, cos_nrm_list = [], []              # new
# mse_nrm_para_list, cos_nrm_para_list = [], []    # new
# nla_correct_list, paraphrase_correct_list = [], []

# from tqdm.auto import tqdm

# for i, ex in enumerate(tqdm(ds, desc="Evaluating", unit="ex")):
#     prompt = ex["question"] + "\nAnswer: Let's think step by step."
#     gold = gsm8k_gold(ex)

#     text_base, h_orig = run_with_patch(prompt, patch_vec=None, max_new_tokens=MAX_NEW_TOKENS)
#     correct["baseline"] += extract_final_answer(text_base) == gold
#     h_orig_list.append(h_orig)

#     explanation = av(h_orig)

#     # Condition 2: plain round-trip
#     h_hat = ar(explanation)
#     h_hat_list.append(h_hat)
#     mse, cos = mse_cos_from_reconstruction(h_hat, h_orig, critic.mse_scale)
#     mse_list.append(mse); cos_list.append(cos)
#     sq_err_list.append(raw_sq_error(h_hat, h_orig))

#     mse_nrm, cos_nrm = critic.score(explanation, h_orig.detach().float().cpu())
#     mse_nrm_list.append(mse_nrm); cos_nrm_list.append(cos_nrm)

#     text_nla, _ = run_with_patch(prompt, patch_vec=h_hat, max_new_tokens=MAX_NEW_TOKENS)
#     is_correct = extract_final_answer(text_nla) == gold
#     nla_correct_list.append(is_correct)
#     correct["nla_roundtrip"] += is_correct

#     # Condition 3: paraphrase round-trip
#     paraphrased = paraphrase(explanation)
#     h_hat_para = ar(paraphrased)
#     h_hat_para_list.append(h_hat_para)
#     mse_p, cos_p = mse_cos_from_reconstruction(h_hat_para, h_orig, critic.mse_scale)
#     mse_para_list.append(mse_p); cos_para_list.append(cos_p)
#     sq_err_para_list.append(raw_sq_error(h_hat_para, h_orig))

#     mse_nrm_p, cos_nrm_p = critic.score(paraphrased, h_orig.detach().float().cpu())
#     mse_nrm_para_list.append(mse_nrm_p); cos_nrm_para_list.append(cos_nrm_p)

#     text_para, _ = run_with_patch(prompt, patch_vec=h_hat_para, max_new_tokens=MAX_NEW_TOKENS)
#     is_correct_para = extract_final_answer(text_para) == gold
#     paraphrase_correct_list.append(is_correct_para)
#     correct["paraphrase_roundtrip"] += is_correct_para

# n = N_SANITY

# fve_roundtrip = fve(sq_err_list, h_orig_list)
# fve_paraphrase = fve(sq_err_para_list, h_orig_list)

# fve_nrm_roundtrip = fve_nrm(sum(mse_nrm_list) / len(mse_nrm_list))            # new
# fve_nrm_paraphrase = fve_nrm(sum(mse_nrm_para_list) / len(mse_nrm_para_list)) # new

# print(f"{'Condition':<22}{'Accuracy':<12}{'Cos-sim':<12}{'MSE':<12}{'FVE':<10}{'FVE (nrm)':<12}")
# print(f"{'Baseline':<22}{correct['baseline']/n:<12.2%}{'--':<12}{'--':<12}{'--':<10}{'--':<12}")
# print(f"{'NLA round-trip':<22}{correct['nla_roundtrip']/n:<12.2%}"
#       f"{sum(cos_list)/len(cos_list):<12.3f}{sum(mse_list)/len(mse_list):<12.4f}"
#       f"{fve_roundtrip:<10.3f}{fve_nrm_roundtrip:<12.3f}")
# print(f"{'Paraphrase round-trip':<22}{correct['paraphrase_roundtrip']/n:<12.2%}"
#       f"{sum(cos_para_list)/len(cos_para_list):<12.3f}{sum(mse_para_list)/len(mse_para_list):<12.4f}"
#       f"{fve_paraphrase:<10.3f}{fve_nrm_paraphrase:<12.3f}")

# %% [markdown]
# Standard reconstruction metrics disagree sharply depending on whether they're scale-normalized: cosine similarity (0.992) indicates the AR recovers the activation's direction well, while raw-scale FVE is strongly negative because 81% of total squared error is concentrated in just 10 of 5,376 dimensions — a small set of "massive activation" dimensions with near-constant, very large magnitude across examples. This suggests the AR is not well-calibrated on these specific outlier dimensions, even though it captures the semantically relevant structure of the activation.

# %%
# labels = ["Baseline", "NLA round-trip", "Paraphrase round-trip"]
# values = [correct["baseline"]/n, correct["nla_roundtrip"]/n, correct["paraphrase_roundtrip"]/n]

# plt.figure(figsize=(5,4))
# plt.bar(labels, values, color=["#888", "#7F77DD", "#DD9E4F"])
# plt.ylabel("Accuracy")
# plt.ylim(0, 1)
# for i, v in enumerate(values):
#     plt.text(i, v - 0.08, f"{v:.0%}", ha="center")
# plt.title(f"GSM8K accuracy (n={n})")
# plt.tight_layout()
# plt.savefig("accuracy_comparison.png", dpi=150)
# plt.show()

# # --- Reconstruction-quality summary: cos-sim, MSE, FVE per condition ---
# # Baseline has no reconstruction, so it's excluded from these three.
# recon_labels = ["NLA round-trip", "Paraphrase round-trip"]
# recon_colors = ["#7F77DD", "#DD9E4F"]

# cos_means = [sum(cos_list)/len(cos_list), sum(cos_para_list)/len(cos_para_list)]
# mse_means = [sum(mse_list)/len(mse_list), sum(mse_para_list)/len(mse_para_list)]
# fve_vals = [fve_roundtrip, fve_paraphrase]

# fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# axes[0].bar(recon_labels, cos_means, color=recon_colors)
# axes[0].set_ylabel("Cosine similarity")
# axes[0].set_ylim(0, 1)
# axes[0].set_title("Mean cos-sim")
# for i, v in enumerate(cos_means):
#     axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center")

# axes[1].bar(recon_labels, mse_means, color=recon_colors)
# axes[1].set_ylabel("MSE")
# axes[1].set_title("Mean MSE")
# for i, v in enumerate(mse_means):
#     axes[1].text(i, v, f"{v:.4f}", ha="center", va="bottom")

# axes[2].bar(recon_labels, fve_vals, color=recon_colors)
# axes[2].set_ylabel("FVE")
# axes[2].set_ylim(0, 1)
# axes[2].set_title("Fraction of variance explained")
# for i, v in enumerate(fve_vals):
#     axes[2].text(i, v + 0.02, f"{v:.3f}", ha="center")

# for ax in axes:
#     ax.tick_params(axis='x', rotation=15)

# plt.suptitle(f"Reconstruction quality by condition (n={n})")
# plt.tight_layout()
# plt.savefig("reconstruction_quality_summary.png", dpi=150)
# plt.show()

# # --- Per-example cos-sim vs correctness, one panel per reconstruction condition ---
# fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

# panel_data = [
#     (cos_list, nla_correct_list, "NLA round-trip"),
#     (cos_para_list, paraphrase_correct_list, "Paraphrase round-trip"),
# ]

# for ax, (cos_vals, correct_flags, title) in zip(axes, panel_data):
#     colors = ["#639922" if c else "#E24B4A" for c in correct_flags]
#     y_jitter = 1 + np.random.uniform(-0.05, 0.05, size=len(cos_vals))
#     ax.scatter(cos_vals, y_jitter, c=colors, s=80, alpha=0.8)
#     ax.set_ylim(0.8, 1.2)
#     ax.set_yticks([])
#     ax.set_xlabel("Cosine similarity (reconstruction vs original)")
#     ax.set_title(title)

# plt.suptitle("Reconstruction quality per example\n(green=correct, red=incorrect)")
# plt.tight_layout()
# plt.savefig("cos_vs_correctness.png", dpi=150)
# plt.show()

# %%
# # Hardcoding due to cut off
# cos_means = [0.992, 0.982]
# mse_means = [0.0164, 0.0361]
# fve_vals = [-10.963, -5.030]

# fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# axes[0].bar(recon_labels, cos_means, color=recon_colors)
# axes[0].set_ylabel("Cosine similarity")
# axes[0].set_ylim(0, 1)
# axes[0].set_title("Mean cos-sim")
# for i, v in enumerate(cos_means):
#     axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center")

# axes[1].bar(recon_labels, mse_means, color=recon_colors)
# axes[1].set_ylabel("MSE")
# axes[1].set_title("Mean MSE")
# for i, v in enumerate(mse_means):
#     axes[1].text(i, v, f"{v:.4f}", ha="center", va="bottom")

# axes[2].bar(recon_labels, fve_vals, color=recon_colors)
# axes[2].set_ylabel("FVE")
# axes[2].set_title("Fraction of variance explained")
# for i, v in enumerate(fve_vals):
#     axes[2].text(i, v + (0.3 if v >= 0 else -0.6), f"{v:.3f}", ha="center")

# for ax in axes:
#     ax.tick_params(axis='x', rotation=15)

# plt.suptitle(f"Reconstruction quality by condition (n={n})")
# plt.tight_layout()
# plt.savefig("reconstruction_quality_summary.png", dpi=150)
# plt.show()

# %% [markdown]
# # Experiment 2 - Tasks from Paper

# %% [markdown]
# ## New Tasks
# 
# We construct a diverse array of over 40 relatively simple tasks to test whether function
# vectors can be extracted in diverse settings. To simplify the presentation of our analysis, we focus on
# a representative sample of 6 tasks:
# - Antonym. Given an input word, generate the word with opposite meaning.
#     - Nguyen et al (2017)
# - Capitalize. Given an input word, generate the same word with a capital first letter.
#     - ChatGPT
# - Country-Capital. Given a country name, generate the capital city.
#     - ChatGPT
# - English-French. Given an English word, generate the French translation of the word.
#     - Conneau et al (2017)
# - Present-Past. Given a verb in present tense, generate the verb’s simple past inflection.
#     - ChatGPT
# - Singular-Plural. Given a singular noun, generate its plural inflection
#     - ChatGPT
# 
# 
# For each task have a plot for number of shots (X axis) and accuracy (y-axis). 

# %%
"""
Shot-count sweep version of the NLA round-trip eval on the 6 ICL
function-vector tasks (Todd et al. 2023, arXiv:2310.15213).

For each task, sweeps the number of ICL demonstrations (x-axis) and
plots accuracy (y-axis) for two conditions:
  - nla_roundtrip:        h patched with AR(verbalize(h))
  - paraphrase_roundtrip: h patched with AR(paraphrase(verbalize(h)))

Also now tracks FVE (fraction of variance explained) for the
reconstructed activation vs. the actual activation, for both
conditions, per (task, n_shots) config.

Produces one PNG per task plus a combined 2x3 grid, and caches raw
results to JSON so you can replot without rerunning the model.

Notebook cells -- paste/import your existing definitions (av,
paraphrase, ar, run_with_patch, mse_cos_from_reconstruction,
raw_sq_error, LastTokenPatcher, client, critic, model,
tokenizer) before running this. (fve / fve_nrm from earlier drafts
are superseded by compute_fve below -- see the note in its docstring
if you're keeping the old per-example versions around too.)

Dataset sources, all expected flat under datasets/:
  - antonym:           Nguyen et al. 2017 (AntSynNET) -- github.com/nguyenkh/AntSynNET
  - capitalize:        ChatGPT-generated
  - country-capital:   ChatGPT-generated
  - english-french:    Conneau et al. 2017 (MUSE bilingual dictionaries) --
                        both en-fr.0-5000.txt (train) and en-fr.5000-6500.txt
                        (test) splits, concatenated into one pool
  - present-past:      ChatGPT-generated
  - singular-plural:   ChatGPT-generated
"""

# %%
import json
import random
import string
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

TASKS = [
    "antonym",
    "capitalize",
    "country-capital",
    "english-french",
    "present-past",
    "singular-plural",
]

# Shot counts to sweep. 0 = zero-shot (no demonstrations, just "Q: <word>\nA:").
N_SANITY = 200
SHOT_COUNTS = [0, 2, 4, 6, 8, 10]
N_PER_CONFIG = 50
MAX_NEW_TOKENS = 256
SEED = 0

DATA_DIR = Path("datasets")
OUT_DIR = Path("/mnt/ssd-2/soar-nla/kaylee")
RESULTS_CACHE = OUT_DIR / "fv_shot_sweep_results.json"
CASE_SENSITIVE_TASKS = {"capitalize"}


# %%
# --------------------------------------------------------------------------
# Per-task dataset loaders -- three different source formats, so three loaders
# --------------------------------------------------------------------------

def load_antonym(data_dir: Path, pos_tags=("verb", "noun", "adjective"),
                  splits=("train", "val", "test"),
                  tokenizer=None) -> list[dict]:
    """AntSynNET (Nguyen et al. 2017), antonym pairs only.

    Files are named like 'verb-pairs.train', 'noun-pairs.val',
    'adjective-pairs.test' -- tab-separated: word1, word2, label,
    where label is 1 for antonym pairs and 0 for synonym pairs.

    Combines all POS categories across all splits into one pool,
    keeps only label==1 rows, de-dupes exact (input, output) pairs,
    and -- if a tokenizer is passed -- further filters to pairs where
    BOTH words tokenize as a single token, matching Todd et al. 2023
    (2,398 antonym pairs kept after that step).
    """
    seen = set()
    pairs = []
    for pos in pos_tags:
        for split in splits:
            path = data_dir / f"{pos}-pairs.{split}"
            if not path.exists():
                print(f"[warn] missing file: {path}")
                continue
            with open(path) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue
                    w1, w2, label = parts[0], parts[1], parts[2]
                    if label != "1":          # keep antonyms only, drop synonyms (0)
                        continue
                    key = (w1, w2)
                    if key in seen:
                        continue
                    seen.add(key)
                    pairs.append({"input": w1, "output": w2})

    if tokenizer is not None:
        def is_single_token(word: str) -> bool:
            return len(tokenizer.encode(" " + word, add_special_tokens=False)) == 1
        pairs = [p for p in pairs
                 if is_single_token(p["input"]) and is_single_token(p["output"])]

    return pairs


def load_english_french(paths) -> list[dict]:
    """Conneau et al. 2017 (MUSE) bilingual dictionary format: one
    'english_word french_word' pair per line, whitespace-separated.
    Accepts a single path or a list of paths -- MUSE ships en-fr as
    separate train (0-5000) / test (5000-6500) splits, but nothing in
    this pipeline trains/evaluates a mapping, so both are concatenated
    into one pool rather than kept separate.

    MUSE dictionaries list multiple valid translations for some English
    words (e.g. 'chat' -> 'discuter'/'discussion'/'chat'/...). Only the
    FIRST translation seen for a given English word is kept -- later
    duplicates are dropped -- so every English word maps to exactly one
    canonical French output. This avoids two problems: (a) the same
    input appearing as both a demonstration and the query with
    conflicting gold answers, and (b) is_correct() marking a model's
    valid-but-different translation as wrong non-deterministically.
    It does NOT fix the underlying scoring looseness -- a model that
    produces a *dropped* valid translation still scores as incorrect.
    If that turns out to matter, switch to scoring against the full
    set of valid translations per word instead of exact-match-one-string.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    seen = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                en, fr = parts[0], parts[1]
                seen.setdefault(en, fr)
    return [{"input": en, "output": fr} for en, fr in seen.items()]


def load_json_pairs(path: Path) -> list[dict]:
    """ChatGPT-generated tasks (capitalize, country-capital, present-past,
    singular-plural). Expects a JSON list of {"input": ..., "output": ...}.
    """
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "examples" in raw:
        raw = raw["examples"]
    return [{"input": ex["input"], "output": ex["output"]} for ex in raw]


# task -> (loader function, path(s) relative to DATA_DIR)
# adjust filenames once you know what your antonym download is actually named
TASK_LOADERS = {
    "antonym": (load_antonym, None),   # loader takes the data_dir directly
    "capitalize": (load_json_pairs, "capitalize.json"),
    "country-capital": (load_json_pairs, "country-capital.json"),
    "english-french": (load_english_french, ["en-fr.0-5000.txt", "en-fr.5000-6500.txt"]),
    "present-past": (load_json_pairs, "present-past.json"),
    "singular-plural": (load_json_pairs, "singular-plural.json"),
}

def load_fv_task(task_name: str, data_dir: Path = DATA_DIR, tokenizer=None) -> list[dict]:
    loader, rel_path = TASK_LOADERS[task_name]
    if task_name == "antonym":
        return loader(data_dir, tokenizer=tokenizer)
    if isinstance(rel_path, list):
        full_paths = [data_dir / p for p in rel_path]
    else:
        full_paths = data_dir / rel_path
    return loader(full_paths)


# %%
# --------------------------------------------------------------------------
# Prompting / scoring
# --------------------------------------------------------------------------

def build_icl_prompt(pairs: list[dict], query_idx: int, n_shots: int, rng: random.Random) -> tuple[str, str]:
    """Excludes candidate demonstrations by matching INPUT WORD, not just
    index -- some datasets (e.g. english-french before dedup, or any task
    with accidental duplicate inputs) can otherwise let the same word
    appear as both a demonstration and the query, or as two demonstrations
    with different gold outputs, which confuses the in-context signal.
    """
    query_word = pairs[query_idx]["input"]
    pool = [i for i in range(len(pairs)) if i != query_idx and pairs[i]["input"] != query_word]
    shot_idxs = rng.sample(pool, min(n_shots, len(pool)))
    lines = [f"Q: {pairs[i]['input']}\nA: {pairs[i]['output']}" for i in shot_idxs]
    lines.append(f"Q: {query_word}\nA:")
    instruction = "Answer with a single word only. No punctuation, no explanation."
    prompt = instruction + "\n\n" + "\n\n".join(lines)
    gold = pairs[query_idx]["output"]
    return prompt, gold


def extract_word_answer(text: str) -> str:
    first_line = text.strip().split("\n")[0]
    return first_line.strip(string.punctuation + " ")


def is_correct(pred: str, gold: str, case_sensitive: bool) -> bool:
    if not case_sensitive:
        pred, gold = pred.lower(), gold.lower()
    return pred == gold


def is_meaningfully_paraphrased(original: str, paraphrased: str, max_overlap: float = 0.7) -> bool:
    orig_words = set(original.lower().split())
    para_words = set(paraphrased.lower().split())
    if not orig_words:
        return True
    overlap = len(orig_words & para_words) / len(orig_words)
    return overlap < max_overlap


# %%
# --------------------------------------------------------------------------
# FVE (fraction of variance explained)
# --------------------------------------------------------------------------

def compute_fve(actual: list[torch.Tensor], reconstructed: list[torch.Tensor]) -> float:
    """FVE for a batch of N reconstructed activation vectors against
    their N actual counterparts, per Pavlos's spec:

        y = actual activation, x = reconstructed activation
        variance = sum_n sum_d (y_n,d - mean_n(y)_d)^2   <- pooled over ALL N points
        FVU      = sum_n sum_d (y_n,d - x_n,d)^2
        FVE      = 1 - FVU / variance

    Both sums run over the full (N, D) batch jointly -- the per-dimension
    mean in the variance term is the mean *across the N examples in this
    config*, not each example's own across-dimension mean. That pooling is
    what makes this a genuine "fraction of variance explained" rather than
    a per-example reconstruction-error ratio, so call this once per
    (task, n_shots, condition) config over all N samples, not once per
    example.

    On normalization: AR's output is raw-scale (it wasn't reparametrized
    to output normalized activations), and per Pavlos, normalization was
    only ever applied on the loss side during AR's training -- i.e. the
    MSE loss AR was optimized against was itself divided by the batch
    variance, which is exactly the FVU/variance ratio here. So computing
    FVE directly on raw y/x and letting the `variance` denominator do the
    normalizing reproduces the same normalized quantity AR was trained
    against, without introducing a second, inconsistent normalization
    (e.g. per-vector unit-norm scaling) on top of it. If you later
    confirm AR's training loss instead normalized each example
    individually (e.g. divided by ||y_n|| before the MSE, rather than by
    the batch variance), switch this to normalize each row of Y and X by
    that same per-example quantity before the sums below -- do it
    identically to both Y and X, or the ratio stops meaning what it says.
    """
    Y = torch.stack([y.detach().float().cpu().reshape(-1) for y in actual])          # (N, D)
    X = torch.stack([x.detach().float().cpu().reshape(-1) for x in reconstructed])   # (N, D)

    y_mean = Y.mean(dim=0, keepdim=True)                 # (1, D), pooled across N
    variance = ((Y - y_mean) ** 2).sum().item()
    fvu = ((Y - X) ** 2).sum().item()

    if variance == 0:
        return float("nan")
    return 1.0 - (fvu / variance)

# %%
# --------------------------------------------------------------------------
# One (task, n_shots) configuration
# --------------------------------------------------------------------------

def run_config(task_name: str, n_shots: int, pairs: list[dict], rng: random.Random, verbose=False) -> dict:
    n = min(N_PER_CONFIG, len(pairs))
    query_idxs = rng.sample(range(len(pairs)), n)
    case_sensitive = task_name in CASE_SENSITIVE_TASKS

    nla_correct = 0
    para_correct = 0
    para_overlaps = []
    para_low_overlap_count = 0

    # Collected for FVE -- one entry per sample, aggregated at the end
    # of the config rather than per-example, per compute_fve's docstring.
    actual_acts = []
    nla_recon_acts = []
    para_recon_acts = []

    for qi in query_idxs:
        prompt, gold = build_icl_prompt(pairs, qi, n_shots, rng)
        _, h_orig = run_with_patch(prompt, patch_vec=None, max_new_tokens=MAX_NEW_TOKENS)
        explanation = av(h_orig)

        if verbose:
            print(f"--- gold={gold!r} ---")
            print("explanation:", explanation[:300])

        # NLA round-trip
        h_hat = ar(explanation)
        text_nla, _ = run_with_patch(prompt, patch_vec=h_hat, max_new_tokens=MAX_NEW_TOKENS)
        pred = extract_word_answer(text_nla)
        correct = is_correct(pred, gold, case_sensitive)
        if verbose:
            print("[nla]  pred:", pred, "| correct:", correct)
        nla_correct += correct

        actual_acts.append(h_orig)
        nla_recon_acts.append(h_hat)

        # Paraphrase round-trip
        paraphrased = paraphrase(explanation)
        orig_words = set(explanation.lower().split())
        para_words = set(paraphrased.lower().split())
        overlap = len(orig_words & para_words) / len(orig_words) if orig_words else 0.0
        para_overlaps.append(overlap)
        if not is_meaningfully_paraphrased(explanation, paraphrased):
            para_low_overlap_count += 1
            if verbose:
                print(f"[para] WARNING: low reword rate (overlap={overlap:.2f})")

        h_hat_para = ar(paraphrased)
        text_para, _ = run_with_patch(prompt, patch_vec=h_hat_para, max_new_tokens=MAX_NEW_TOKENS)
        pred_para = extract_word_answer(text_para)
        correct_para = is_correct(pred_para, gold, case_sensitive)
        if verbose:
            print("[para] paraphrased:", paraphrased[:300])
            print("[para] pred:", pred_para, "| correct:", correct_para)
        para_correct += correct_para

        para_recon_acts.append(h_hat_para)

    return {
        "n_shots": n_shots,
        "n": n,
        "acc_nla": nla_correct / n,
        "acc_para": para_correct / n,
        "fve_nla": compute_fve(actual_acts, nla_recon_acts),
        "fve_para": compute_fve(actual_acts, para_recon_acts),
        # just to make sure
        "avg_para_overlap": sum(para_overlaps) / n,
        "para_low_overlap_count": para_low_overlap_count,
    }


# %%
# --------------------------------------------------------------------------
# Sweep + plotting
# --------------------------------------------------------------------------
#
# One DataFrame per task, saved to its own CSV the moment that task
# finishes -- not batched up into one big dict/JSON at the end. If task 4
# of 6 throws (OOM, a bad dataset file, whatever), tasks 1-3 are already
# on disk as fv_shot_sweep_<task>.csv and load_sweep_results() below can
# pull them back in for plotting without rerunning anything. The sweep
# also catches per-task exceptions and moves on to the next task rather
# than aborting the whole run.

from datetime import datetime

# %%
# --------------------------------------------------------------------------
# Sweep + plotting
# --------------------------------------------------------------------------

import pandas as pd


def run_sweep(run_id: str | None = None) -> dict[str, pd.DataFrame]:
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    rng = random.Random(SEED)
    results: dict[str, pd.DataFrame] = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for task_name in TASKS:
        try:
            pairs = load_fv_task(task_name)
            rows = [
                run_config(task_name, k, pairs, rng, verbose=True)
                for k in tqdm(SHOT_COUNTS, desc=task_name, unit="shot-count")
            ]
        except Exception as e:
            print(f"[error] task {task_name!r} failed, skipping it and continuing: {e!r}")
            continue

        df = pd.DataFrame(rows)
        results[task_name] = df

        csv_path = OUT_DIR / f"fv_shot_sweep_{task_name}_{run_id}.csv"
        try:
            df.to_csv(csv_path, index=False)
            print(f"[ok] {task_name} -> {csv_path}")
        except OSError as e:
            print(f"[warn] ran {task_name} but could not save its CSV: {e}")

    return results, run_id


def load_sweep_results(run_id: str, tasks: list[str] = TASKS) -> dict[str, pd.DataFrame]:
    """Reload whatever per-task CSVs already exist on disk for a given
    run_id -- e.g. after a kernel crash mid-sweep, or if you just want to
    replot without rerunning the model. Silently skips tasks that never
    finished/saved.
    """
    results: dict[str, pd.DataFrame] = {}
    for task_name in tasks:
        csv_path = OUT_DIR / f"fv_shot_sweep_{task_name}_{run_id}.csv"
        if csv_path.exists():
            results[task_name] = pd.read_csv(csv_path)
        else:
            print(f"[warn] no cached CSV for {task_name} at {csv_path}")
    return results


def plot_sweep(results: dict[str, pd.DataFrame], run_id: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for task_name, df in results.items():
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(df["n_shots"], df["acc_nla"], marker="o", label="NLA round-trip")
        ax.plot(df["n_shots"], df["acc_para"], marker="s", label="Paraphrase round-trip")
        ax.set_xlabel("Number of shots")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.set_title(task_name)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"fv_shot_sweep_{task_name}_{run_id}.png", dpi=150)
        plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, task_name in zip(axes.flat, TASKS):
        if task_name not in results:
            ax.set_title(f"{task_name} (no data)")
            ax.axis("off")
            continue
        df = results[task_name]
        ax.plot(df["n_shots"], df["acc_nla"], marker="o", label="NLA round-trip")
        ax.plot(df["n_shots"], df["acc_para"], marker="s", label="Paraphrase round-trip")
        ax.set_title(task_name)
        ax.set_xlabel("Number of shots")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    handled_axes = [ax for ax in axes.flat if ax.lines]
    if handled_axes:
        handled_axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"fv_shot_sweep_all_tasks_{run_id}.png", dpi=150)
    plt.show()


results, run_id = run_sweep()
plot_sweep(results, run_id)
print(f"This run's ID: {run_id}")

# If a run gets interrupted partway, skip run_sweep() entirely next time
# and instead do:
#   results = load_sweep_results()
#   plot_sweep(results)
# to replot everything that made it to disk.

# %% [markdown]
# - might have to loosen to check if word is there or not instead of expecting exact output
# - The av/ar round-trip through natural language seems to preserve the subject matter far more reliably than it preserves what specifically to do with that subject — the fine-grained target word, or the task's input→output mapping. That's a meaningful and fairly interpretable finding for whatever you're writing up, distinct from just "accuracy is low."

# %% [markdown]
# ## Cleanup
# 
# Close the AV client's HTTP session and stop SGLang only if this notebook launched it. An AV server that was already running before the notebook is left untouched.
# 

# %%
if "client" in globals() and hasattr(client, "_http"):
    client._http.close()
    print("Closed NLA client HTTP session")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

if "sglang_proc" in globals() and sglang_proc is not None:
    sglang_proc.terminate()
    sglang_proc.wait(timeout=30)
    print("Stopped SGLang launched by this notebook")
else:
    print("No SGLang subprocess launched by this notebook")

# Optional final memory check:
# !nvidia-smi

# %%



