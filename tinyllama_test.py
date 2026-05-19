import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "Explain what an LLM jailbreak attack is in one paragraph."

messages = [
    {
        "role": "user",
        "content": prompt
    }
]

formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    formatted_prompt,
    return_tensors="pt"
).to(model.device)

print("Generating response...")

outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    temperature=0.7
)

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\n=== MODEL RESPONSE ===\n")
print(response)
