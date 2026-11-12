# Promping

## Types

### Direct prompts

Explicit instructions

Example:

"What's the capital of Benin"

Use case: for simple, factual queries

### Open-Ended prompts

Encourage creative or explinatory responses

Example:

"Tell me a story about a dragon and a knight"

Use case: creative writing, brainstorming, writing a craft of a story...

### Instructional prompts

Detailed instructions for an specific task

Examples:

"Write a 300-word summary of the benefits of exercise, using bullet points"

Use case: Structured task like summarization or analysis

### Role-base prompts

Assigning a role or persona to the model

Example:

"You are a historian. Explain the causes of World War I."

## Principles of effective prompts engineering

Clarity, Specificity, Contextualization, Iterative Refinement and Leverage Examples

## Techniques

### Few-Shot promping

Provide few examples in the prompt to guide the model

Example:

"Translate this senctences: 'Hello' -> 'Hola', 'Goodbye' -> 'Adiós'. Now translate: 'Thank you'

### Zero-shot promping

Ask the model to perform a task without examples.

Example:

"Translate 'Good morning' into Spanish'

### Chain-of-Thought promping

Encourage the model to break down complex task into steps.

Example:

"Solve this problem step by step: if John has 5 apples and gives 2 to Mary, how many does he have left?"

### Role promping

Assign a role to the model to tailor its responses.

Example:

"You are a career coach. Give advice to someone switching careers."
"Act as a career coach. Give advice to someone switching careers."

### Temp and Top-p Sampling

Adjust parameters like temperature (creativity) and top-p (diversity) to control output randomness.

Example:

Use low temperature for factual responses (temperature close to 0 or 0) and high temperature for creative writing (temperature close to 1 or 1)

```python
completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a creative writer."},
        {"role": "user", "content": "Write a creative tagline for a coffee shop."},
    ],
    # Do not change both at the same time
    # temperature=0.9,  # controls the randomness of the output
    top_p=0.9,  # controls the diversity of the output
)
# print(completion.choices[0].message.content)
```
