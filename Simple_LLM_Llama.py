from transformers import LlamaForCausalLM, LlamaTokenizer

def generate_text(prompt, max_length=50, token=None):
    # Define the model name
    model_name = "meta-llama/Meta-Llama-3.1-8B"  # Ensure this is the correct model name
    
    # Load pre-trained model and tokenizer with the provided token
    model = LlamaForCausalLM.from_pretrained(model_name, use_auth_token=token)
    tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=token)
    
    # Tokenize the input text
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate text
    outputs = model.generate(inputs['input_ids'], max_length=max_length, num_return_sequences=1)
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

# Example usage with token
prompt = "How was GPT-2 model trained"
token = "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"  # Replace with your actual token
generated_txt = generate_text(prompt, token=token)
print(generated_txt)
