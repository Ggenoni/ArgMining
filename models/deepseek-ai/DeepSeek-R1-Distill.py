from transformers import AutoModelForCausalLM, AutoTokenizer

# Choose the model you want to load
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # or "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

prompt = "Suggest 3 critical questions before accepting this argument: 'AI will replace all jobs.'"

inputs = tokenizer(prompt, return_tensors="pt").to(device)
output = model.generate(**inputs, max_length=100)

# Decode and print output
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
