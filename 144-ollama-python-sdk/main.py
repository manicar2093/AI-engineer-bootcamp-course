import ollama

response = ollama.list()
# print(response)

response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": "why is the sky blue"}])
print(response["message"]["content"])

print("\n\n\nStreaming response:")
response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": "why ocean is salty"}], stream=True)
for chunk in response:
    print(chunk["message"]["content"], end="", flush=True)


print("\n\n\nShow model information:")
response = ollama.show(model="llama3.2")
print(response)
