import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
from PIL import Image
import io
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)
from trl import SFTConfig, SFTTrainer

# ==========================================
# 1. Load Tokenizers & Processors
# ==========================================
print("Loading processors...")

# Cosmos-Reason2 processor (handles image + text interleaving)
model_name = "nvidia/Cosmos-Reason2-2B"
llm_processor = AutoProcessor.from_pretrained(model_name)

# FAST Action Tokenizer (converts continuous robotic actions to discrete tokens)
# fast_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
fast_tokenizer = AutoProcessor.from_pretrained('/home/nboehme/Desktop/NVIDIA_Hackathon_2026/cosmos-reason2/examples/notebooks/fast_tokenizer_2', trust_remote_code=True)
# fast_tokenizer = AutoProcessor.from_pretrained('/home/nboehme/Desktop/NVIDIA_Hackathon_2026/cosmos-reason2/examples/notebooks/fast_tokenizer', trust_remote_code=True)
print(f"Tokenizer type: {type(fast_tokenizer)}")
fast_vocab_size = fast_tokenizer.vocab_size if hasattr(fast_tokenizer, 'vocab_size') else 4096
new_action_tokens = [f"<action_{i}>" for i in range(fast_vocab_size)]

# 2. Add them to the LLM's tokenizer
num_added_toks = llm_processor.tokenizer.add_tokens(new_action_tokens)
print(f"Added {num_added_toks} new action tokens to the processor.")

# The id for "assistant". Used for masking.
assistant_token_id = llm_processor.tokenizer.convert_tokens_to_ids("assistant")

# ==========================================
# 2. Load and Preprocess the Dataset
# ==========================================
print("Loading Libero dataset...")
# Using a small slice for demonstration purposes. Change to split="train" for full training.
dataset = load_dataset("physical-intelligence/libero", split="train[:1%]") 
# Full Dataset
# dataset = load_dataset("physical-intelligence/libero", split="train") 
ep_ids = list(dataset["episode_index"])
# Find where the episode ID changes
end_of_episode_ind = [ep_ids.index(i + 1) for i in range(ep_ids[-1])]
end_of_episode_ind.append(len(ep_ids))
all_acts = np.array(dataset["actions"])
ACTION_CHUNK_SIZE = 50  # Prediction steps into the future
action_dim = all_acts.shape[-1]

def prepare_actions(batch, indices):
    chunked_output = np.zeros((len(indices), ACTION_CHUNK_SIZE, action_dim))
    # Access the episode_indices for the entire dataset
    for i, idx in enumerate(indices):
        available = min(end_of_episode_ind[ep_ids[idx]] - idx, ACTION_CHUNK_SIZE)
        real_actions = all_acts[idx : idx + available]
        chunked_output[i, :available] = real_actions
        
        # Fast padding: Repeat the last available action for the rest of the chunk
        if available < ACTION_CHUNK_SIZE:
            chunked_output[i, available:] = real_actions[-1]

    # Tokenize the now-safe chunks
    action_tokens_batch = fast_tokenizer(chunked_output)

    # Transform integers into special token strings
    # Example: 124 -> "<action_124>"
    action_strs = []
    for tokens in action_tokens_batch:
        token_str = " ".join([f"<action_{t}>" for t in tokens])
        action_strs.append(token_str)
        
    return {"action_str": action_strs}

print("Tokenizing action chunks...")
train_dataset = dataset.map(
    prepare_actions, 
    batched=True, 
    batch_size=32,
    with_indices=True # Necessary to look up future actions in the global dataset
)

def custom_vla_collator(features):
    texts = []
    images = []
    
    for feature in features:
        # 1. Safely grab the raw PIL image
        images.append(feature["image"])
        images.append(feature["wrist_image"])
        action_str = feature["action_str"]
        
        # 2. Build the conversational structure for the text prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"}, # The template just needs the type to insert the placeholder
                    {"type": "image"}, # TODO: Is this adding the wrist image?
                    {"type": "text", "text": "Given the current visual observations, what action sequence should the robotic arm take?"}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": action_str}
                ]
            }
        ]
        
        # 3. Apply the chat template to get the raw string (with <|image_pad|> tokens)
        text = llm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
        
    # 4. Let the processor handle tokenization and image tensor generation together
    batch = llm_processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True
    )
    
    # 5. Create labels for SFT (copy input_ids and mask out padding tokens)
    labels = batch["input_ids"].clone()
    for i, input_ids in enumerate(batch["input_ids"]):
        # Mask all tokens up until the assistant starts
        # Note: Replace <|assistant|> with the actual tag for Cosmos-Reason2
        assistant_start_idx = (input_ids == assistant_token_id).nonzero(as_tuple=True)[0]
        if len(assistant_start_idx) > 0:
            labels[i, :assistant_start_idx[0] + 1] = -100 
        else:
            print("WARNING: Could not find assistant token for masking. Prompt is not being masked!")

    # Mask pad tokens
    pad_token_id = llm_processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = llm_processor.tokenizer.eos_token_id
        
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
        
    batch["labels"] = labels
    
    return batch

# ==========================================
# 3. Load Model with QLoRA Configuration
# ==========================================
print("Loading quantized Cosmos-Reason2 model...")

# Configure 4-bit quantization to fit the model in consumer GPUs
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    quantization_config=bnb_config,
)
# 3. Resize the model embeddings to match the new tokenizer length
model.resize_token_embeddings(len(llm_processor.tokenizer))
# Make sure gradients pass through
model.enable_input_require_grads()
model.get_input_embeddings().requires_grad_(True)
model.gradient_checkpointing_enable()
model.config.use_cache = False

# Configure LoRA adapter
peft_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=[
        "down_proj",
        "o_proj",
        "k_proj",
        "q_proj",
        "gate_proj",
        "up_proj",
        "v_proj",
    ],
    modules_to_save=["embed_tokens", "lm_head"],
    task_type="CAUSAL_LM",
    ensure_weight_tying=True,
)

# ==========================================
# 4. Configure Training
# ==========================================
output_dir = "outputs/Cosmos-Reason2-VLA-Libero-Chunked"

training_args = SFTConfig(
    max_steps=500, 
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=8, 
    warmup_steps=5,
    learning_rate=2e-4,
    optim="adamw_8bit", 
    max_length=512, # 2048
    remove_unused_columns=False, # Keep the 'image' column so our collator can see it
    dataset_kwargs={"skip_prepare_dataset": True}, # We are handling preparation in the collator
    output_dir=output_dir,
    logging_dir=f"{output_dir}/logs",
    logging_steps=8,
    save_steps=25,
    report_to=["tensorboard"],
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config,
    data_collator=custom_vla_collator,
    processing_class=llm_processor,
)

# ==========================================
# 5. Train and Save
# ==========================================
print("Starting training...")

trainer_stats = trainer.train()

print(f"Training complete. Saving adapter to {output_dir}")
trainer.save_model(output_dir)
llm_processor.save_pretrained(output_dir)
