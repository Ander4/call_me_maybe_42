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


def test2() -> None:

    model = Small_LLM_Model()
    vocab = model.get_path_to_vocab_file()

    with open(vocab, encoding="utf-8") as file:
        data = json.load(file)

    filtered_list = filter_digits(data)
    stop_id = [data["."]]
    space_id = data["Ġ"]
    allowed = filtered_list + stop_id + [space_id]

    sentence = model.encode("My password is ")
    input_ids = sentence[0].tolist()
    answer = []

    while (True):
        logits = model.get_logits_from_input_ids(input_ids)
        masked = mask_logits(logits, allowed)
        best = int(np.argmax(masked))

        print(f"Best: '{model.decode([best])}'")

        if best in stop_id:
            break

        input_ids.append(best)
        answer.append(best)

    decoded_text = model.decode(input_ids)
    print(f"Texto generado: {decoded_text}")
    decoded_answer = model.decode(answer)
    inteo = int(decoded_answer)
    print(f"Respuesta generada: '{inteo}', Type: {type(inteo)}")


def main() -> None:
    model = Small_LLM_Model()
    vocab = model.get_path_to_vocab_file()

    with open(vocab, encoding="utf-8") as file:
        data = json.load(file)

    sentence = model.encode("What is the sum of 2 and 3? Call the function")
    input_ids = sentence[0].tolist()

    candidate1 = model.encode("fn_add_numbers")
    candidate2 = model.encode("fn_greet")
    candidate3 = model.encode("fn_reverse_string")

    candidate1_ids = candidate1[0].tolist()
    print(f"Candidate1: {candidate1_ids}")
    candidate2_ids = candidate2[0].tolist()
    print(f"Candidate2: {candidate2_ids}")
    candidate3_ids = candidate3[0].tolist()
    print(f"Candidate3: {candidate3_ids}")

    alive = [candidate1_ids, candidate2_ids, candidate3_ids]
    step = 0
    generated = []
    stopper = data['"']
    while True:
        allowed = []
        helper = []
        for candidate in alive:
            if step == len(candidate):
                if stopper not in allowed:
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
            if candidate[step] == best:
                helper.append(candidate)

        alive = helper
        step += 1
        if len(alive) == 1 and step == len(alive[0]):
            decoded_text = model.decode(generated)
            print(f"Texto generado: {decoded_text}")
            break


if __name__ == "__main__":
    main()

"vocab path: /sgoinfre/students/aeiros-t/students/aeiros-t/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
