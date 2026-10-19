from llm_sdk.llm_sdk import Small_LLM_Model
import json


def mask_logits(logits: list[float], allowed: list[int]) -> list[float]:
    """Devuelve una copia de logits con -inf salvo en los IDs permitidos."""

    masked = [
        float("-inf")] * len(logits)
    for token_id in allowed:
        masked[token_id] = logits[token_id]
    return masked


def allow_all_except(
        logits: list[float],
        prohibited: list[int]) -> list[float]:

    masked = list(logits)
    for token_id in prohibited:
        masked[token_id] = float("-inf")

    return masked


def filter_digits(data: dict[str, int]) -> list[int]:

    filtered: list[int] = []
    for k, v in data.items():
        if k.isdecimal():
            filtered.append(v)

    return filtered


def build_forbidden_tokens(model: Small_LLM_Model) -> list[int]:

    tokenizer = model.get_path_to_tokenizer_file()

    prohibited: list[int] = []
    with open(tokenizer, encoding="utf-8") as file:
        tokenizer_data = json.load(file)

    added_tokens_id = [element["id"]
                       for element in tokenizer_data["added_tokens"]]

    new_line_id = model.encode("\n")[0].tolist()[0]
    tab_id = model.encode("\t")[0].tolist()[0]

    prohibited = added_tokens_id + [new_line_id, tab_id]

    return prohibited
