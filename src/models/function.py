from pydantic import BaseModel, model_validator
from .param_type import ParamType


class Function(BaseModel):
    """A class that contains all the information of a function.

    Attributes:
        name (str): the name of the function.
        description (str): a description of the function.
        parameters (dict[str, ParamType]): a dictionary containing each
            parameter name and its type.
        returns (ParamType): the type of the return of the function.
    """

    name: str
    description: str
    parameters: dict[str, ParamType]
    returns: ParamType

    @model_validator(mode="after")
    def validate_values(self) -> "Function":
        if not self.name:
            raise ValueError("function name can not be empty")
        if not self.description:
            raise ValueError("function description can not be empty")
        return self

    def print(self) -> None:
        print(f"Name: {self.name}\n"
              f"Description: {self.description}\n"
              f"Parameters: {self.parameters}\n"
              f"Returns: {self.returns}")
