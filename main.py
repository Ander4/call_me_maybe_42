""" from src.parser.Parser import Parser
import sys """
from llm_sdk import Small_LLM_Model
import numpy as np
import json


def tests() -> None:

    model = Small_LLM_Model()

    sentence = model.encode("The capital of France is")
    input_ids = sentence[0].tolist()

    for i in range(1, 8):
        logits = model.get_logits_from_input_ids(input_ids)
        index = np.argmax(logits)
        input_ids.append(int(index))

    print(f"Los input_ids: {input_ids}")
    decoded_text = model.decode(input_ids)
    print(f"Texto generado: {decoded_text}")

    """ parser = Parser()
    with open(sys.argv[1], "r") as calling_input:
        print(sys.argv[1] + ":")
        for line in calling_input.readlines():
            parser.parse_line(line)

    print("\n")

    with open(sys.argv[2], "r") as functions_input:
        print(sys.argv[2] + ":")
        for line in functions_input.readlines():
            parser.parse_line(line) """


def mask_logits(logits: list[float], allowed: list[int]) -> list[float]:
    """Devuelve una copia de logits con -inf salvo en los IDs permitidos."""
    masked = [
        float("-inf")] * len(logits)
    for token_id in allowed:
        masked[token_id] = logits[token_id]
    return masked


def filter_digits(vocab: dict[str, int]) -> list[int]:

    filtered: list[int] = []
    for k, v in vocab.items():
        if k.isdecimal():
            filtered.append(v)

    return filtered


def main() -> None:

    model = Small_LLM_Model()
    vocab = model.get_path_to_vocab_file()

    with open(vocab, encoding="utf-8") as file:
        data = json.load(file)

    filtered_list = filter_digits(data)
    print(f"Len: {len(filtered_list)}, values:\n{filtered_list}")

    sentence = model.encode("The number of days in a week is ")
    input_ids = sentence[0].tolist()
    logits = model.get_logits_from_input_ids(input_ids)
    masked = mask_logits(logits, filtered_list)
    stopper = dict[","]
    print(f"Stopper: '{stopper}'")
    best = int(np.argmax(masked))
    print(model.decode([best]))

    input_ids.append(int(best))
    decoded_text = model.decode(input_ids)
    print(f"Texto generado: {decoded_text}")


if __name__ == "__main__":
    main()
