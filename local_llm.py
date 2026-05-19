import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LocalLLM:

    def __init__(self, model_name):

        self.model_name = model_name

        print(f"\nLoading model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        print("Model loaded successfully.")

    def query(self, prompt, max_new_tokens=150):

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7
        )

        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return response
