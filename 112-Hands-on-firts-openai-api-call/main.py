from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

res = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a mexican poet living on 1890 in a poor neighborhood in Mexico City."},
        {"role": "user", "content": "Write a poem about the meaning of life"},
    ]
)

print(res.choices[0].message.content)
