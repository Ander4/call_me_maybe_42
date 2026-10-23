from enum import Enum
from pydantic import BaseModel


class ParamTypeName(str, Enum):
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"


class ParamType(BaseModel):
    """An auxiliary class that contains the type of a parameter.

    Attributes:
        type (ParamTypeName): the type of the parameter, which can only be:
            - number
            - string
            - boolean.
    """
    type: ParamTypeName
