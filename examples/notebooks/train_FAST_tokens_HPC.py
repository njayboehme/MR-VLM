import os
import importlib.util
import numpy as np
from datasets import load_dataset
from huggingface_hub import snapshot_download

# ----------------------------
# Config
# ----------------------------
ACTION_CHUNK_SIZE = 50
BASE_FAST_REPO = "physical-intelligence/fast"
OUTPUT_DIR = "/TODO/TODO/fast_tokenizer"
VOCAB_SIZE = 1024
# ----------------------------
# Load FAST processor manually
# ----------------------------
repo_path = snapshot_download(BASE_FAST_REPO)

proc_file = os.path.join(repo_path, "processing_action_tokenizer.py")
spec = importlib.util.spec_from_file_location("processing_action_tokenizer", proc_file)
mod = importlib.util.module_from_spec(spec)

# THIS IS KEY!!!!!!!!!!!!!
import sys
sys.modules["processing_action_tokenizer"] = mod # Register the module globally

spec.loader.exec_module(mod)

from transformers import PreTrainedTokenizerFast

bpe_tokenizer = PreTrainedTokenizerFast.from_pretrained(repo_path)

tokenizer = mod.UniversalActionProcessor(
    bpe_tokenizer=bpe_tokenizer,
    scale=10,
    vocab_size=VOCAB_SIZE,
    min_token=-354,
)
# THIS IS KEY!!!!!!!!!!!!!!!!!!!
tokenizer.register_for_auto_class("AutoProcessor")

# ----------------------------
# Load dataset
# ----------------------------
dataset = load_dataset("physical-intelligence/libero", split="train")

all_acts = np.asarray(dataset["actions"], dtype=np.float32)
ep_ids = np.asarray(dataset["episode_index"])

num_steps, action_dim = all_acts.shape
chunked_output = np.zeros((num_steps, ACTION_CHUNK_SIZE, action_dim), dtype=np.float32)

# ----------------------------
# Compute episode end indices robustly
# ----------------------------
episode_end = {}
for idx in range(num_steps - 1):
    if ep_ids[idx + 1] != ep_ids[idx]:
        episode_end[ep_ids[idx]] = idx + 1
episode_end[ep_ids[-1]] = num_steps

# ----------------------------
# Build fixed-length chunks
# ----------------------------
for i in range(num_steps):
    ep = ep_ids[i]
    end = episode_end[ep]
    available = min(end - i, ACTION_CHUNK_SIZE)

    real_actions = all_acts[i : i + available]
    chunked_output[i, :available] = real_actions

    if available < ACTION_CHUNK_SIZE:
        chunked_output[i, available:] = real_actions[-1]

# ----------------------------
# Train tokenizer
# ----------------------------
tokenizer = tokenizer.fit(
    chunked_output,
    vocab_size=VOCAB_SIZE,
    time_horizon=ACTION_CHUNK_SIZE,
    action_dim=action_dim,
)

# ----------------------------
# Save tokenizer
# ----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Save the tokenizer
tokenizer.save_pretrained(OUTPUT_DIR)