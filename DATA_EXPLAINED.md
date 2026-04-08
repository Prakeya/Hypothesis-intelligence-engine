# Understanding the Fact Matrix (JSON)

To test a hypothesis, the Engine requires **Evidence**. In The Alchemist Lab, this evidence is provided in a format called **JSON** (JavaScript Object Notation).

## Why is it required?
Imagine trying to prove that "Higher temperatures melt more ice cream" without any actual weather or sales data. You would just be guessing. 

The Engine is designed to be a **Logic Judge**. It doesn't guess; it calculates. By providing a JSON dataset, you are feeding the Engine a "Truth Matrix"—a structured list of facts it can use to verify your claim.

## How to format it?
The Engine looks for a list of data points. Each point should have a name (key) and a value.

### Simple Example:
If you want to test if **Coffee reduces Sleep**, your data might look like this:

```json
[
  {"cups": 0, "sleep_hours": 8},
  {"cups": 2, "sleep_hours": 6},
  {"cups": 4, "sleep_hours": 4}
]
```

### Key Rules:
1. **The Wrapper**: Use square brackets `[ ]` to start and end your list.
2. **The Data Points**: Each point is inside curly braces `{ }`.
3. **The Connection**: Use a colon `:` between the name and the value.
4. **The Separator**: Use a comma `,` between different data points.

By providing this "Digital Blueprint," you allow the AI to see the mathematical shape of reality, ensuring every conclusion is grounded in logic rather than hallucination.
