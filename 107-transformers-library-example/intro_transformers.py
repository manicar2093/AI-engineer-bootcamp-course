from transformers import pipeline, GenerationConfig

def create_simple_llm():
    model_name = "distilgpt2"

    generator = pipeline(
        'text-generation',
        model=model_name,
        generation_config=GenerationConfig(
            pad_token_id=50256,
            max_new_tokens=1000,
            num_return_sequences=1
        ),
    )
    return generator


generator = create_simple_llm()

prompt = "Once upon a time"

generated_text = generator(prompt)
res = generated_text[0]['generated_text']

print(res)
