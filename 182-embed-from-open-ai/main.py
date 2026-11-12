from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="The Milky Way galaxy is estimated to contain 100-400 billion stars.",
)

print(response.data[0].embedding)
