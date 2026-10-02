""" from src.parser.Parser import Parser
import sys """
from llm_sdk import Small_LLM_Model
import numpy as np
import json


def tests() -> None:

    model = Small_LLM_Model()

    prompt = "Greet the user"
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n</think>\n"
    sentence = model.encode(formatted_prompt)
    input_ids = sentence[0].tolist()
    print(f"Los input_ids: {input_ids}")

    for i in range(30):
        logits = model.get_logits_from_input_ids(input_ids)
        index = int(np.argmax(logits))
        input_ids.append(index)

    decoded_text = model.decode(input_ids)
    print(f"Texto generado: '{decoded_text}'")

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


def test2() -> None:

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


def main() -> None:
    model = Small_LLM_Model()
    vocab = model.get_path_to_vocab_file()

    with open(vocab, encoding="utf-8") as file:
        data = json.load(file)

    prompt = "Greet the user"
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n</think>\n"
    sentence = model.encode(formatted_prompt)
    input_ids = sentence[0].tolist()

    candidate1 = model.encode("fn_add_numbers")
    candidate2 = model.encode("fn_greet")
    candidate3 = model.encode("fn_reverse_string")
    candidate4 = model.encode("fn_greet_all")

    candidate1_ids = candidate1[0].tolist()
    print(f"Candidate1: {candidate1_ids}")
    candidate2_ids = candidate2[0].tolist()
    print(f"Candidate2: {candidate2_ids}")
    candidate3_ids = candidate3[0].tolist()
    print(f"Candidate3: {candidate3_ids}")
    candidate4_ids = candidate4[0].tolist()
    print(f"Candidate4: {candidate4_ids}")

    print(model.decode([2891]))   # candidato de "add_numbers"
    print(model.decode([1889]))   # candidato de "greet"
    print(model.decode([43277]))  # candidato de "reverse_string"

    alive = [candidate1_ids, candidate2_ids, candidate3_ids, candidate4_ids]
    step = 0
    generated = []
    stopper = data['"']
    while True:

        print(f"========== Vuelta {step} ==========")

        allowed = []
        helper = []
        for candidate in alive:
            if step == len(candidate):
                allowed.append(stopper)
            elif candidate[step] not in allowed:
                allowed.append(candidate[step])
        print(f"Permitidos: {allowed}")

        logits = model.get_logits_from_input_ids(input_ids)
        masked = mask_logits(logits, allowed)
        best = int(np.argmax(masked))

        input_ids.append(best)
        generated.append(best)

        print(f"Best: {best}")

        for candidate in alive:
            if step < len(candidate):
                if candidate[step] == best:
                    helper.append(candidate)

        alive = helper
        step += 1
        print(f"Alive this turn: {alive}")
        print("========== Fin Vuelta ==========")
        if best == stopper:
            decoded_text = model.decode(generated)
            print(f"Texto generado: {decoded_text}")
            break


if __name__ == "__main__":
    main()
