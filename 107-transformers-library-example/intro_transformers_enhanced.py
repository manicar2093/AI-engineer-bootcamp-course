from transformers import pipeline, AutoTokenizer, GenerationConfig


def create_simple_llm(max_length: int=1000):
    model_name = "distilgpt2"

    generator = pipeline(
        'text-generation',
        model=model_name,
        generation_config=GenerationConfig(
            pad_token_id=50256,
            max_length=max_length,  # max length of the text generated
            num_return_secuence=1, # number of different sequences generated
            do_sample=True, # use sampling vs greedy decoding
            temperature=0.7, # randomness of the sampling (0 = deterministic, 1 = random)
        ),
    )
    return generator


def generate_text(generator, prompt):
    """
    Generate text based on a prompt
    """
    # Generate text
    result = generator(
        prompt,  # input text
    )

    return result[0]["generated_text"]


# Educational demonstration
def run_llm_demo():
    """
    Demonstrates basic LLM functionality with explanations
    """
    print("🤖 Loading Simple LLM Model...")
    generator = create_simple_llm()

    print("\n✨ Simple LLM Demo ✨")
    print("This demo shows basic text generation using a small language model")

    # Example prompts to demonstrate different capabilities
    prompts = [
        "The quick brown fox",
        "Once upon a time",
        "Python programming is",
    ]

    for prompt in prompts:
        print("\n🔹 Prompt:", prompt)
        print("🔸 Generated:", generate_text(generator, prompt))
        input("\nPress Enter to see next example...")


# Interactive demo
def interactive_demo():
    """
    Allows users to interact with the model
    """
    generator = create_simple_llm()

    print("\n🤖 Interactive LLM Demo")
    print("Type your prompts (or 'quit' to exit)")

    while True:
        prompt = input("\n✍️ Enter your prompt: ")
        if prompt.lower() == "\\q":
            break

        response = generate_text(generator, prompt)
        print("\n💭 Generated response:")
        print(response)


# Educational visualization of the process
def explain_process():
    """
    Explains the LLM process with a simple example
    """
    print("\n🎓 How it works:")
    print("1. Input text → Tokenization → Numbers")
    print("2. Numbers → Model Processing → Prediction")
    print("3. Prediction → New Token → Output Text")

    # Simple tokenization example
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    text = "Hello world!"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    print("\n📝 Example Tokenization:")
    print(f"Original text: '{text}'")
    print(f"As tokens (numbers): {tokens}")
    print(f"Decoded back: '{decoded}'")


if __name__ == "__main__":
    print("Choose a demo:")
    print("1. Run basic demonstration")
    print("2. Interactive mode")
    print("3. Explain the process")

    choice = input("Enter your choice (1-3): ")

    if choice == "1":
        run_llm_demo()
    elif choice == "2":
        interactive_demo()
    elif choice == "3":
        explain_process()
    else:
        print("Invalid choice!")
