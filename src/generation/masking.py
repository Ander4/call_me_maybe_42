from llm_sdk.llm_sdk import Small_LLM_Model
import json


def mask_logits(logits: list[float], allowed: list[int]) -> list[float]:
    """A function that returns a copy of the logits, with -inf as value except
    for the allowed IDs.

    Args:
        logits (list[float]): the raw logits for the next token.
        allowed (list[int]): a list with the allowed IDs.

    Returns:
        list[float]: a list with all the logits at -inf value except for the
            allowed IDs.
    """

    masked = [
        float("-inf")] * len(logits)
    for token_id in allowed:
        masked[token_id] = logits[token_id]
    return masked


def allow_all_except(
        logits: list[float],
        prohibited: list[int]) -> list[float]:
    """A function that returns a copy of the logits, with -inf as value to the
    IDs that are not allowed.

    Args:
        logits (list[float]): the raw logits for the next token.
        prohibited (list[int]): a list with the prohibited IDs.

    Returns:
        list[float]: a list that has the prohibited IDs as -inf.
    """

    masked = list(logits)
    for token_id in prohibited:
        masked[token_id] = float("-inf")

    return masked


def filter_digits(data: dict[str, int]) -> list[int]:
    """A function that returns a list with the IDs of every digit from 0 to 9.

    Args:
        data(dict[str, int]): the vocabulary from the model, in a dictionary
            format.

    Returns:
        list[int]: a list containing the IDs of every digit from 0 to 9.
    """

    filtered: list[int] = []
    for k, v in data.items():
        if k.isdecimal():
            filtered.append(v)

    return filtered


def build_forbidden_tokens(model: Small_LLM_Model) -> list[int]:
    """A function that returns a list with the IDs that are prohibited for the
    model.

    Args:
        model (Small_LLM_Model): the LLM model.

    Returns:
        list[int]: a list containing the IDs of the prohibited tokens.
    """

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
