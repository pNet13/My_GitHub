from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer

def generate_textgpt(prompt, max_length=100):
    """
    Generate text using GPT-2 model.
    
    Parameters:
    - prompt (str): The input text prompt for text generation.
    - max_length (int): The maximum length of the generated text.
    
    Returns:
    - str: The generated text.
    """
    # Load pre-trained model and tokenizer
    model_name = "gpt2-large"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Set padding token to be the same as eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize input prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # Generate text
    outputs = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        num_return_sequences=1,
        attention_mask=inputs.get('attention_mask'),  # Attention mask if available
        pad_token_id=tokenizer.pad_token_id  # Ensure padding is handled correctly
    )

    # Decode generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

# Example usage
# This exmaple can be extented for other texts completiong exercises

if __name__ == "__main__":
    prompt = "How was GPT-2 model trained"
    generated_txt = generate_textgpt(prompt)
    print(generated_txt)
