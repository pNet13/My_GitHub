from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load GPT-2 model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Ensure the model is in evaluation mode
model.eval()

def generate_response(prompt, model, tokenizer, max_new_tokens=100):
    # Encode the prompt text
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate a response
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,  # Use sampling instead of greedy decoding
            temperature=0.7  # Control randomness; lower is less random
        )
    
    # Decode the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Initialize conversation
current_input = "Give me math to descripe theory of everything"
max_exchanges = 3
exchange_count = 0

while exchange_count < max_exchanges:
    print(f"Model 1: {current_input}")
    
    # Model 1 generates a response
    model1_response = generate_response(current_input, model, tokenizer)
    
    # Print Model 1's response
    print(f"Model 1 Response: {model1_response}")
    
    # Model 2 takes Model 1's response as input
    model2_response = generate_response(model1_response, model, tokenizer)
    
    # Print Model 2's response
    print(f"Model 2 Response: {model2_response}")
    
    # Prepare for the next exchange
    current_input = model2_response
    exchange_count += 1
