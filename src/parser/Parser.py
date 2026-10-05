#!/usr/bin/env python3
import json
from typing import Any
from pydantic import ValidationError, BaseModel


class Parser:

    def __init__(self) -> None:
        pass

    def parse_input_file(self, path: str) -> list[dict[str, Any]]:

        try:
            with open(path, encoding="UTF-8") as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Exploto el parser por cosas de archivo:\n{e}")
            exit(1)
        return data

    def parse_items(
            self, data: list, model: type[BaseModel]) -> list[BaseModel]:

        try:
            functions = [model(**item) for item in data]
        except ValidationError as e:
            print(f"Error validando functions_definition.json: {e}")
            exit(1)

        return functions
