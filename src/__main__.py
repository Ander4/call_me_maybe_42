from llm_sdk.llm_sdk import Small_LLM_Model
import json
from .parser.parser import Parser
from .models import Prompt, Function
from .generation import (PromptGen, DataGenerator, convert_param_value)
import argparse
from typing import NoReturn


class ArgParser(argparse.ArgumentParser):
    """Customizes ArgumentParser's error formatting.

    Subclass of argparse.ArgumentParser that overrides error() to print
    a shorter, personalized error message.
    """

    def error(self, message: str) -> NoReturn:

        print(self.format_usage())
        print(f"error: {message}")
        exit(2)


def generate_output_json(
        output: list[dict[str, str | dict[str, int | float | str]]],
        path: str) -> None:
    """A function that transforms the given list into a JSON and
    saves it into the given file

    Args:
        output (list[dict[str, str | dict[str, int | float | str]]]): the
            generated output by the model
        path (str): the path of the file in which the JSON will be saved
    """

    try:
        with open(path, "w", encoding="UTF-8") as file:

            json.dump(output, file, indent=4)
            print(
                "Succesfully created the output in the following file:"
                f"\n'{path}'")
    except FileNotFoundError:
        print(
            "The folder 'output' doesn't exist when trying to write into "
            f"'{path}'.")
        exit(2)
    except PermissionError:
        print(f"You don't have permission to write into the file '{path}'.")
        exit(2)


def validate_filenames(args: argparse.Namespace) -> None:
    """A function that validates that the given files have the correct .json
    extension.

    Args:
        args (argparse.Namespace): the created pared arguments which contains
            the file's names.
    """

    files: list[str] = [args.functions_definition, args.input, args.output]
    for file in files:
        if (not file.endswith(".json")):
            print(f"Error: the file '{file}' isn't a .json file")
            exit(1)


def create_argparser() -> argparse.Namespace:
    """A function that parses the inputted arguments

    Return:
            Namespace: the created parsed arguments
    """

    parser = ArgParser(prog="uv run python -m src")
    parser.add_argument(
        "--functions_definition", type=str,
        default="functions_definition.json")
    parser.add_argument("--input", type=str,
                        default="function_calling_tests.json")
    parser.add_argument("--output", type=str,
                        default="output.json")

    return parser.parse_args()


def main() -> None:

    parser = Parser()

    input_path = "data/input/"
    output_path = "data/output/"
    args = create_argparser()
    validate_filenames(args)
    functions_file = input_path + args.functions_definition
    prompts_file = input_path + args.input
    output_file = output_path + args.output

    try:
        functions_data = parser.parse_input_file(functions_file)
        functions_list = parser.parse_items(
            functions_data, Function, functions_file)

        prompts_data = parser.parse_input_file(prompts_file)
        prompts_list = parser.parse_items(prompts_data, Prompt, prompts_file)
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        # print(f"Error parsing input files: {e}")
        print(e)
        exit(1)

    model = Small_LLM_Model()
    vocab = model.get_path_to_vocab_file()

    with open(vocab, encoding="utf-8") as file:
        data = json.load(file)

    candidate_tokens = [model.encode(function.name)[0].tolist()
                        for function in functions_list]

    output_list: list[dict[str, str | dict[str, int | float | str]]] = []

    prompt_gen = PromptGen(functions_list, model)
    data_gen = DataGenerator(model, data)
    for prompt in prompts_list:

        dictionary_to_add: dict[str, str | dict[str, int | float | str]] = {}
        input_ids = prompt_gen.get_input_ids(prompt)

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

            parameters_result[param_name] = convert_param_value(
                param_type, generated, model)

        dictionary_to_add["prompt"] = prompt.prompt
        dictionary_to_add["name"] = output_function.name
        dictionary_to_add["parameters"] = parameters_result
        output_list.append(dictionary_to_add)

    generate_output_json(output_list, output_file)


if __name__ == "__main__":
    main()
