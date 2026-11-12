import ollama

model_name = "ocean-assistant"

ollama.create(
    model=model_name,
    from_="llama3.2",
    system="You are a very smart assistant who knows everythin about oceans. You are very succint and informative.",
    parameters={
        "temperature": 0.1,
    },
)

res = ollama.generate(model=model_name, prompt="why ocean is so salty")
print(res["response"])

ollama.delete(model=model_name)
