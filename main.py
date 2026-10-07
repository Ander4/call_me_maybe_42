from llm_sdk import Small_LLM_Model
from pydantic import BaseModel
import json
from src.parser.Parser import Parser
from src.models import Prompt, Function
import numpy as np


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


def build_forbidden_tokens(
        model: Small_LLM_Model, data: dict[str, int]) -> list[int]:

    tokenizer = model.get_path_to_tokenizer_file()

    prohibited: list[int] = []
    with open(tokenizer, encoding="utf-8") as file:
        tokenizer_data = json.load(file)

    added_tokens_id = [element["id"]
                       for element in tokenizer_data["added_tokens"]]
    new_line_id = model.encode("\n")[0].tolist()[0]
    tab_id = model.encode("\t")[0].tolist()[0]

    prohibited = [data['"']] + added_tokens_id + [new_line_id, tab_id]

    return prohibited


def build_functions_info(functions_list: list[Function]) -> str:
    functions_info = ""
    for function in functions_list:
        functions_info += f"\n- {function.name}: {function.description}"

    return functions_info


def build_chat_prompt(functions_info: str, user_prompt: Prompt,
                      model: Small_LLM_Model) -> list[int]:
    formatted_prompt = (
        f"<|im_start|>user\nAvailable functions:{functions_info}\n\n"
        f"Question: {user_prompt.prompt}<|im_end|>\n"
        "<|im_start|>assistant\n</think>\n"
    )
    sentence = model.encode(formatted_prompt)
    input_ids = sentence[0].tolist()

    return input_ids


def generate_answer(alive: list[list[int]], stopper: int, input_ids: list[int],
                    model: Small_LLM_Model) -> tuple[list[int], list[int]]:

    ids_copy = list(input_ids)
    step = 0
    generated: list[int] = []
    while True:

        allowed = []
        helper = []
        for candidate in alive:
            if step == len(candidate):
                allowed.append(stopper)
            elif candidate[step] not in allowed:
                allowed.append(candidate[step])

        logits = model.get_logits_from_input_ids(ids_copy)
        masked = mask_logits(logits, allowed)
        best = int(np.argmax(masked))

        ids_copy.append(best)
        generated.append(best)

        for candidate in alive:
            if step < len(candidate):
                if candidate[step] == best:
                    helper.append(candidate)

        alive = helper
        step += 1
        if best == stopper:
            break

    return generated, ids_copy


def main() -> None:

    parser = Parser()

    functions_file = "/sgoinfre/students/aeiros-t/Call_Me/call_me_maybe_42/data/input/test_functions.json"
    prompts_file = "/sgoinfre/students/aeiros-t/Call_Me/call_me_maybe_42/data/input/test_input.json"

    functions_data = parser.parse_input_file(functions_file)
    functions_list = parser.parse_items(functions_data, Function)

    prompts_data = parser.parse_input_file(prompts_file)
    prompts_list = parser.parse_items(prompts_data, Prompt)

    # Hasta aqui seria bloque 1: Manejo de archivos(Tambien se decidiria aqui la ruta del output que de momento no he hecho)

    model = Small_LLM_Model()
    vocab = model.get_path_to_vocab_file()

    with open(vocab, encoding="utf-8") as file:
        data = json.load(file)

    build_forbidden_tokens(model, data)

    functions_info = build_functions_info(functions_list)

    i = 0

    candidate_tokens = [model.encode(function.name)[0].tolist()
                        for function in functions_list]

    for prompt in prompts_list:

        input_ids = build_chat_prompt(functions_info, prompt, model)

        alive = list(candidate_tokens)

        stopper = data['"']

        generated, input_ids = generate_answer(
            alive, stopper, input_ids, model)

        decoded_text = model.decode(generated)
        print(f"Texto generado para el prompt {i}: {decoded_text}")
        i += 1

        clean_output = decoded_text.removesuffix('"')
        print(f"Cleaned output: {clean_output}")

        output_function: Function | None = None
        for function in functions_list:
            if function.name == clean_output:
                output_function = function
                break

        if output_function is None:
            print(f"Aviso: no se reconoció la función '{clean_output}' "
                  f"para el prompt: '{prompt.prompt}'. Se omite este prompt.")
            continue

        to_add = ', "parameters": {"'
        for param in output_function.parameters.keys():
            if not_last:
                to_add += f'{param}":' + generar_valor + ', "'
            else:
                to_add += f'{param}":' + generar_valor + '}'
        print(f"Lo que hay que añadir: {to_add}")


if __name__ == "__main__":
    main()
