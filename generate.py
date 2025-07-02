from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

prompt = "Once upon a time in a land far away,"

# Encode input
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# Generate with temperature = 0.7
output_0_7 = model.generate(
    input_ids,
    max_length=50 + len(input_ids[0]),
    do_sample=True,
    top_k=50,
    temperature=0.7
)
text_0_7 = tokenizer.decode(output_0_7[0], skip_special_tokens=True)

# Generate with temperature = 1.0
output_1_0 = model.generate(
    input_ids,
    max_length=50 + len(input_ids[0]),
    do_sample=True,
    top_k=50,
    temperature=1.0
)
text_1_0 = tokenizer.decode(output_1_0[0], skip_special_tokens=True)

# Save to file
with open("samples.txt", "w") as f:
    f.write("## Temperature = 0.7\n")
    f.write(text_0_7 + "\n\n")
    f.write("## Temperature = 1.0\n")
    f.write(text_1_0)
