from llm_sdk.llm_sdk import Small_LLM_Model
import json
from .parser.Parser import Parser
from .models import Prompt, Function, ParamType
from .generation import (build_forbidden_tokens, PromptGen, DataGenerator)


def clean_parameter_output(
        generated: list[int],
        param_type: ParamType, model: Small_LLM_Model) -> str:

    decoded_text = model.decode(generated)
    match param_type.type:
        case "number":
            clean_output = decoded_text.removesuffix(',')
        case "string":
            splitted = decoded_text.split('"')
            clean_output = splitted[0]
        case "boolean":
            clean_output = decoded_text

    return clean_output


def convert_param_value(
        text: str, param_type: ParamType) -> float | str | bool:

    return_value: float | str | bool
    match param_type.type:
        case "number":
            return_value = float(text)
        case "string":
            return_value = text
        case "boolean":
            return_value = bool(text)

    return return_value


def generate_output_json(
        output: list[dict[str, str | dict[str, int | float | str]]]) -> None:

    path = "data/output/function_calling_results.json"
    try:
        with open(path, "w", encoding="UTF-8") as file:

            json.dump(output, file, indent=4)
            print(
                "Succesfully created the output in the following file:"
                f"\n'{path}'")

    except FileNotFoundError as e:
        print(f"An error ocurred when trying to create the output file:\n{e}")
        exit(1)


def main() -> None:

    parser = Parser()

    functions_file = "data/input/functions_definition.json"
    prompts_file = "data/input/function_calling_tests.json"

    functions_data = parser.parse_input_file(functions_file)
    functions_list = parser.parse_items(functions_data, Function)

    prompts_data = parser.parse_input_file(prompts_file)
    prompts_list = parser.parse_items(prompts_data, Prompt)

    model = Small_LLM_Model()
    vocab = model.get_path_to_vocab_file()

    with open(vocab, encoding="utf-8") as file:
        data = json.load(file)

    build_forbidden_tokens(model)

    candidate_tokens = [model.encode(function.name)[0].tolist()
                        for function in functions_list]

    output_list: list[dict[str, str | dict[str, int | float | str]]] = []

    prompt_gen = PromptGen(functions_list, model)
    data_gen = DataGenerator(model, data)
    for prompt in prompts_list:

        dictionary_to_add: dict[str, str | dict[str, int | float | str]] = {}
        input_ids = prompt_gen.build_chat_prompt(prompt)

        alive = list(candidate_tokens)

        stopper = data['"']

        generated, input_ids = data_gen.generate_answer(
            alive, stopper, input_ids)

        decoded_text = model.decode(generated)

        clean_output = decoded_text.removesuffix('"')

        output_function: Function | None = None
        for function in functions_list:
            if function.name == clean_output:
                output_function = function
                break

        if output_function is None:
            print(f"WARNING: unknown function '{clean_output}' "
                  f"for the prompt: '{prompt.prompt}'. "
                  "This prompt will be ommited.")
            continue

        parameters_result: dict[str, int | float | str] = {}

        for index, (param_name, param_type) in enumerate(
                output_function.parameters.items()):
            if index == 0:
                to_add = f', "parameters": {{"{param_name}": '
            else:
                to_add = f', "{param_name}": '

            if param_type.type == "string":
                to_add += '"'

            tokens_to_add = model.encode(to_add)[0].tolist()
            input_ids.extend(tokens_to_add)

            generated, input_ids = data_gen.generate_parameter(
                param_type, input_ids)

            cleaned_output = clean_parameter_output(
                generated, param_type, model)
            parameters_result[param_name] = convert_param_value(
                cleaned_output, param_type)

        dictionary_to_add["prompt"] = prompt.prompt
        dictionary_to_add["name"] = output_function.name
        dictionary_to_add["parameters"] = parameters_result
        output_list.append(dictionary_to_add)

    generate_output_json(output_list)


if __name__ == "__main__":
    main()
