from src.parser.Parser import Parser
import sys
from llm_sdk.llm_sdk import Small_LLM_Model


def main() -> None:
    print("Corriendo programa...")
    model = Small_LLM_Model()
    print(model._model_name)

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


if __name__ == "__main__":
    main()
