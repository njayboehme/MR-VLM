import numpy as np
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoTokenizer
)
from tqdm import tqdm
# First, we download the tokenizer from the Hugging Face model hub
# Here, we will not use the pre-trained tokenizer weights, but only the source code
# to train a new tokenizer on our own data.
tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)

# Load your action data for tokenizer training
# Chunks do not need to be of the same length, we will use dummy data
# action_data = np.random.rand(4000, 50, 14)
ACTION_CHUNK_SIZE = 50
dataset = load_dataset("physical-intelligence/libero", split="train") 
action_data = dataset["actions"]
# 1. Pre-calculate the 'distance' to the next episode boundary for the whole dataset
ep_ids = list(dataset["episode_index"])
# Find where the episode ID changes
end_of_episode_ind = [ep_ids.index(i + 1) for i in range(ep_ids[-1])]
end_of_episode_ind.append(len(ep_ids))

# indices = np.array(indices)
all_acts = np.array(dataset["actions"]) # Pre-load once if memory allows, or keep as dataset object

# Pre-allocate output: (Batch, Chunk_Size, Action_Dim)
action_dim = all_acts.shape[-1]
chunked_output = np.zeros((len(action_data), ACTION_CHUNK_SIZE, action_dim))

for i in range(len(action_data)):
    # Determine how many real steps we can take
    y = ep_ids[i]
    z = end_of_episode_ind[ep_ids[i]]
    available = min(end_of_episode_ind[ep_ids[i]] - i, ACTION_CHUNK_SIZE)
    
    # Slice the real actions
    real_actions = all_acts[i : i + available]
    chunked_output[i, :available] = real_actions
    
    # Fast padding: Repeat the last available action for the rest of the chunk
    if available < ACTION_CHUNK_SIZE:
        chunked_output[i, available:] = real_actions[-1]

# Train the new tokenizer, depending on your dataset size this can take a few minutes
# TODO: Can change vocab size if we want
tokenizer = tokenizer.fit(chunked_output, 
                          vocab_size=1024,
                          time_horizon=ACTION_CHUNK_SIZE, 
                          action_dim=len(action_data[0]))

# Save the new tokenizer, 
tokenizer.save_pretrained("/home/nboehme/Desktop/NVIDIA_Hackathon_2026/cosmos-reason2/examples/notebooks/fast_tokenizer")