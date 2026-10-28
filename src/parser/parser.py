#!/usr/bin/env python3
import json
from typing import Any, TypeVar
from pydantic import ValidationError, BaseModel
from src.errors import (InputFileNotFoundError,
                        InvalidJSONError, InvalidDataError)

T = TypeVar("T", bound=BaseModel)


class Parser:
    """Class that parses the input files and its contents."""

    def __init__(self) -> None:
        pass

    def parse_input_file(self, path: str) -> list[dict[str, Any]]:
        """A function that checks wether the given file exists and has a valid
        JSON format.

        Attributes:
            path (str): the path of the file

        Return:
            list[dict[str, Any]]: the content of the JSON file as a list of
                dictionaries
        """

        try:
            with open(path, encoding="UTF-8") as file:
                data = json.load(file)
        except FileNotFoundError:
            raise InputFileNotFoundError(path)
        except json.JSONDecodeError as e:
            raise InvalidJSONError(path, e.doc, e.pos)
        return data

    def parse_items(
            self, data: list[dict[str, Any]],
            model: type[T],
            path: str) -> list[T]:
        """A function that checks wether the given content is valid.

        Attributes:
                data (list[dict[str, Any]]): the extracted content of the JSON
                    file
                model (type[T]): the class to which the data will be parsed to.
                    It can be either the Function or the Prompt class.
        """

        functions: list[T] = []
        for index, item in enumerate(data):
            try:
                functions.append(model(**item))
            except ValidationError as e:
                raise InvalidDataError(path, index, e)

        return functions
