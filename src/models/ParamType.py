from enum import Enum
from pydantic import BaseModel


class ParamTypeName(str, Enum):
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"


class ParamType(BaseModel):
    type: ParamTypeName
