*This project has been created as part of the 42 curriculum by aeiros-t.*

# Call me maybe

# Description

This project takes requests(prompts) written in natural human language and transforms them into structured functions using a small language model(Qwen/Qwen3-0.6B). Given a prompt like "What is the sum of 2 and 3?" and a list of available functions, the program produces an answer, given in a JSON object.

Small models aren't reliable when generating a valid JSON by themselves. Instead of trusting the model to create a valid JSON answer and retrying the process in case of error, this project uses **constrained decoding** to guarantee a 100% valid JSON. In each step of the generation, the troublesome tokens are masked, leaving the model with only tokens that would make a valid answer, with a valid function name and parameters with the right typing.


# Instructions

## Requisites

- Python 3.10 or later
- The given llm_sdk package (the small LLM model)
- Python dependencies: numpy, pydantic
- Development tools: flake8, mypy

## Installation

To install the dependencies just run the ``make install`` command.

## Execution

To run the program you only need to run the ``make run`` command. There are two ways in which you can run the command:

```bash
make run
```

Using this command the program will run with the default files:

- **data/input/functions_definition.json** for the JSON with the available functions.
- **data/input/function_calling_tests.json** for the JSON containing the prompts.
- **data/output/output.json** for the file that will contain the generated JSON.

The other option is to run the program with custom files:

```bash
make run ARGS="[--functions_definition <function_definition_file>] [--input <input_file>] [--output <output_file>]"
```

The program can be run with three optional flags that indicate the path of the files we want to use:
- **–-functions_definition** for the JSON with the available functions.
- **-–input** for the JSON containing the prompts.
- **–-output** for the file that will contain the generated JSON.

The three flags are optional, which means that you can run the program with any number of them. For example you may only want to specify the output file that will be generated but leave the other two as default:

```bash
make run ARGS="--output generated_results.json"
```

This command will run the program with the default files for function definitions and inputs, but save the generated output in the given 'generated_results.json' file.

The files must follow these rules:
- The input files (functions_definition and input) must be inside the ``data/input/`` folder and must exist.
- The output file has to be inside the ``data/output/`` folder.
- The ``data/output/`` directory must exist.
- If the output exists, it must be editable, so that the program can override it.
- If the output file doesn't exist, the program will create it inside the ``data/output/`` folder.

## File formats

The ``functions_definition.json`` must be a list of functions, each one containing a ``name``, ``description``, ``parameters`` and ``returns``. The ``parameters`` must be a map containing each parameter name and type.

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {"a": {"type": "number"},"b": {"type": "number"}},
    "returns": {"type": "number"}
  }
]
```

The ``function_calling_tests.json`` must be a list with ``prompt``.

```json
[
  {
    "prompt": "What is the sum of 2 and 3?"
  },
  {
    "prompt": "What is the sum of 265 and 345?"
  }
]
```

The ``output.json`` will be a list containing the generated answer for each given prompt. Each answer will contain a ``prompt``, ``name`` and ``parameters``.

```json
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 2.0,
            "b": 3.0
        }
    },
    {
        "prompt": "What is the sum of 265 and 345?",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 265.0,
            "b": 345.0
        }
    }
]
```

# Algorithm Explanation

The system relies entirely on **constrained decoding**: at every generation step, the model's logits are masked before sampling, so only tokens that keep the output syntactically and semantically valid can be chosen. No step depends on the model spontaneously producing correct JSON from a prompt alone.

Two masking strategies are used depending on how large the valid token set is:

- **Allow-list masking**: used when the valid set is small and closed (digits, function name candidates, boolean `true`/`false`). Every logit is set to `-inf` except the ones explicitly allowed.
- **Deny-list masking**: used for string parameters, where almost any token is valid. Instead of enumerating the entire vocabulary as allowed, only a handful of forbidden tokens (quotes, special tokens, newlines, tabs) are masked out — far more efficient.

In both cases, generation proceeds the same way at each step: compute the logits for the next token, apply the relevant mask, pick the highest-scoring token among the remaining candidates, append it, and repeat until a stop condition specific to what is being generated is met.

## 1. Prompt generation

The user's natural language request is wrapped in the model's expected chat format, together with the list of available functions and their descriptions, so the model has the information it needs to choose between them. The model's reasoning mode is explicitly skipped to keep generation within the time budget, rather than letting it produce a free-form reasoning block before answering.

**Example.** Given the prompt `"What is the sum of 2 and 3?"` and the three example functions (`fn_add_numbers`, `fn_greet`, `fn_reverse_string`), the text actually fed to the model looks like this:

```
<|im_start|>user
Available functions:
- fn_add_numbers: Add two numbers together and return their sum.
- fn_greet: Generate a greeting message for a person by name.
- fn_reverse_string: Reverse a string and return the reversed result.

Question: What is the sum of 2 and 3?<|im_end|>
<|im_start|>assistant
</think>
```

The trailing `</think>` is added by the program itself, not generated by the model, so generation starts directly on the answer instead of a reasoning block.

## 2. Function generation

The name of the function to call is generated under an allow-list mask: all candidate function names are tokenized up front, and at each step only the tokens that keep the generated text a valid prefix of at least one remaining candidate are allowed. Candidates that stop matching are dropped as generation progresses, until a single one remains and is confirmed complete.

**Example.** With `fn_greet` and `fn_greet_all` both available, both candidates share the same first tokens (`fn_greet`). Once that much has been generated, `fn_greet` is already a complete, valid function name, while `fn_greet_all` still needs more tokens. At that exact step, the allowed set contains two options: a stop token (which would select `fn_greet` as the final answer) and the next token of `fn_greet_all` (which would continue toward the longer name). The model chooses between them based on the prompt's context, this is the only point where more than one candidate is still alive and the decision carries real weight, every step before it, there is no ambiguity (every candidate agrees on the same next token), and every step after it (once only one candidate survives) is fully determined.

## 3. Parameter generation

Once a function is selected, its parameters are generated one at a time, in order. Before each parameter's value, the fixed JSON scaffolding (its name and the opening syntax expected for its type) is added to the context so the model only has to generate the value itself. The masking strategy and stop condition used depend on the parameter's declared type (`number`, `string`, `boolean`), ensuring the generated value always matches the expected type.

**Example.** For `fn_add_numbers`, with parameters `a` and `b` (both `number`), the program builds the scaffolding one parameter at a time:

1. After the function name, the text `, "parameters": {"a": ` is added to the context, and the model generates digits under the allow-list mask until a stop token is produced — e.g. `2`.
2. The text `, "b": ` is added next, and the same process generates the second value — e.g. `3`.

The model never has to produce any of the surrounding JSON syntax itself — only the two digit sequences `2` and `3`.

## 4. Output cleanup and formatting

Each generated value is cleaned of the scaffolding tokens used to detect its end, then converted to its final Python type according to the parameter's declared type. The resulting values are assembled into the output structure (`prompt`, `name`, `parameters`) and collected for every successfully processed prompt before being written to the output file.

**Example.** For the `fn_add_numbers` case above, the raw generated text for parameter `a` includes its stop token, e.g. `"2,"` — the trailing comma is stripped before converting the remaining text to a Python `float` (`2.0`). For a `string` parameter, the raw generated text includes its closing quote, e.g. `Shrek"` — everything up to that quote is kept and used as the Python `str` value (`"Shrek"`), with the quote itself discarded.

## Design Decisions

- **Classes vs. plain functions**: a class is only introduced when it holds real state reused across multiple calls. `PromptGen` precomputes the functions description once; `DataGenerator` precomputes the model reference, vocabulary, and forbidden-token list once. Stateless logic (`masking.py`, `param_conversion.py`) stays as plain functions — wrapping them in a class would add no value.

- **String termination detection by decoded content, not token ID**: the tokenizer fuses common sequences into single tokens (e.g. `"}\n` as one token). Comparing the generated token's ID against a fixed stop ID misses these fused cases, so termination is instead detected by checking whether the *decoded text* of the generated token contains a `"`.

- **Opening quote is forced by the program, not generated by the model**: without this, the model's first token after the parameter prefix can be a fused "space+quote" (an opening quote), which the stop condition misidentifies as the closing quote, producing empty strings. Forcing the opening quote guarantees the only quote the model can generate afterward is the closing one.

- **No explicit `max_steps` safety limit on parameter generation**: tested extensively with deliberately ambiguous prompts (no concrete data to extract). Once the quote-ambiguity bug(explained later) was fixed, no case of uncontrolled generation was observed. Adding a limit was considered unnecessary complexity without evidence it was needed; documented here as a conscious trade-off rather than an oversight.

- **No `case _` fallback in the type-dispatch logic**: `ParamType.type` is a pydantic `Enum` with exactly three members (`number`, `string`, `boolean`). Pydantic already rejects any other value when the input JSON is parsed, so a defensive fallback branch would be unreachable dead code.

## Challenges Faced

- **Quote-ambiguity**: initially, the model's first generated token for a string parameter could be a fused "space+opening-quote" token. Since the stop condition only checked "does this token contain a quote", it stopped immediately, producing empty values. Fixed by having the program itself emit the opening quote as part of the fixed prompt text, so the model only ever needs to generate the closing one.

- **Fused tokens breaking exact-ID comparisons**: the BPE tokenizer sometimes merges a stop character with surrounding content into a single token (e.g. `"}\n`). Comparing `best_token_id == stop_token_id` failed silently in these cases. Solved by decoding the generated token and checking for the target character in its text instead of comparing IDs.

- **Context-dependent token IDs**: the same character or word tokenizes differently depending on surrounding context (e.g. a space before it, or not). Every stop/special token ID used in the project was verified by encoding the actual text in its real context, never assumed from a standalone lookup.

- **`vocab.json` vs. `tokenizer.json`**: `vocab.json` only contains ~151.6k "regular" BPE tokens; special tokens (`<|im_start|>`, `</think>`, etc.) are listed separately in `tokenizer.json` under `added_tokens` and had to be loaded from there to build the forbidden-token list for strings.

## Known Limitations

- String values are assumed not to contain a literal `"` character before the closing one; cleanup takes everything before the first quote found.

- A backslash (`\`) is currently unrestricted inside generated strings and could in theory produce an invalid JSON escape sequence; not observed in practice during testing.

- Only the standalone tabulation token is forbidden inside strings; a tabulation fused into a larger token (alongside a letter) is not filtered.

- No explicit token-count safety limit is enforced during parameter generation (see Design Decisions above).

## Performance Analysis

- 100% valid, JSON output across all test runs — structure and types are guaranteed by construction through constrained decoding, never left to the model's unconstrained output.

- Function selection and argument extraction accuracy was high across both the example test files and custom adversarial prompts (ambiguous wording,
  missing information, multi-parameter functions, mixed parameter types), consistent with the 90%+ target. Occasional imprecise *content* (e.g. a placeholder-like value for a completely uninformative prompt) was observed, but never invalid structure or type.

- Typical generation stays well within the 5-minute processing budget for the provided test sets, thanks to the `</think>` bypass and the deny-list masking strategy for strings (avoiding a 150k-token scan per step).

## Testing Strategy

- Manual testing with the example `functions_definition.json` and `function_calling_tests.json` provided in the project description.

- Additional custom test files covering: functions with multiple parameters of mixed types, deliberately ambiguous prompts with no concrete data to extract, prompts with embedded quotes/punctuation, and boolean parameters.

- Step-by-step debug logging (temporarily enabled during development) was used to inspect the token stream at each generation step, which is how the quote-ambiguity and fused-token bugs described above were found and diagnosed.

## Example Usage

Using the default input files (`data/input/functions_definition.json` and `data/input/function_calling_tests.json`) and default output path:

```bash
uv sync
uv run python -m src
```

Specifying custom input/output files:

```bash
uv run python -m src --functions_definition data/input/functions_definition.json \
                      --input data/input/function_calling_tests.json \
                      --output data/output/output.json
```

Equivalent shortcuts are available through the provided `Makefile`:

```bash
make install
make run
make run ARGS="--functions_definition functions_definition.json --input function_calling_tests.json --output output.json"
```

# Resources

## References
- Pydantic documentation: https://docs.pydantic.dev
- NumPy documentation: https://numpy.org/doc/
- mypy documentation: https://mypy.readthedocs.io
- flake8 and PEP 8: https://flake8.pycqa.org, https://peps.python.org/pep-0008/
- Claude: https://claude.com/es

## AI usage

During this project AI has been used for different purposes:
- **Understanding how the LLM works and how to communicate with it**: Since this was my first project using a LLM model, I asked the AI to give me simple exercises to learn how to communicate with the model(how tokenization works, how to pass from a prompt to tokens, how to decode what was generated, etc.).
- **Code review**: I asked the AI to review my code to find any potential bugs, or any type hint that I may have missed once my own review of the code was done.
- **Documentation**: The AI has also helped me with the formatting of this README and with the grammar review of it.
- **Makefile**: I had the AI review the Makefile of the project, to review inconsistencies and any improvements that could be made.

All core algorithm design choices (masking strategy, candidate-based function selection, parameter generation order, stop-condition handling) were made and implemented by me. AI was not asked to design or write the core algorithm from scratch; most guidance was in the form of questions, conceptual explanations, and small isolated examples, with the implementation and integration done by me.