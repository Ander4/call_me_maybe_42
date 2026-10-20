from pydantic import BaseModel
from .ParamType import ParamType


class Function(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParamType]
    returns: ParamType

    def print(self) -> None:
        print(f"Name: {self.name}\n"
              f"Description: {self.description}\n"
              f"Parameters: {self.parameters}\n"
              f"Returns: {self.returns}")
