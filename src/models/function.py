from pydantic import BaseModel
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

    def print(self) -> None:
        print(f"Name: {self.name}\n"
              f"Description: {self.description}\n"
              f"Parameters: {self.parameters}\n"
              f"Returns: {self.returns}")
