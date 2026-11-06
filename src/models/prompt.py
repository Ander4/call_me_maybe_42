from pydantic import BaseModel, model_validator


class Prompt(BaseModel):
    """An auxiliary class, used for parsing, that contains the prompt from the
    user.

    Attributes:
        prompt (str): the prompt from the user.
    """
    prompt: str

    @model_validator(mode="after")
    def validate_values(self) -> "Prompt":
        if not self.prompt:
            raise ValueError("prompt can not be empty")

        return self
