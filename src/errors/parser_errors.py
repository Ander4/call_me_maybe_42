#!/usr/bin/env python3
from json import JSONDecodeError
from pydantic import ValidationError


class InputFileNotFoundError(FileNotFoundError):
    """Exception for when an input file isn't found

    Attributes:
        file (str): the specific file that wasn't found
    """

    def __init__(self, file: str) -> None:
        super().__init__(f"The file '{file}' doesn't exist.")


class InvalidJSONError(JSONDecodeError):
    """Exception for when an input file doesn't have a valid JSON format

    Attributes:
        file (str): the specific file that is invalid
        doc (str): The JSON document being parsed
        pos (int): The start index of doc where the parsing failed
    """

    def __init__(self, file: str, doc: str, pos: int) -> None:
        super().__init__(
            f"The file '{file}' isn't a valid JSON file", doc, pos)


class InvalidDataError(Exception):
    """Exception for when the data of a JSON is invalid

    Attributes:
        file (str): the specific file that is invalid
        doc (str): The JSON document being parsed
        pos (int): The start index of doc where the parsing failed
    """

    def __init__(self, file: str, index: int, error: ValidationError) -> None:

        msg = f"Parsing error on file '{file}':\nMissing fields for element {
            index}:\n"
        for element in error.errors():
            msg = msg + "- " + element['loc'][0] + "\n"
        super().__init__(msg)
