from pydantic import BaseModel


class Prompt(BaseModel):
    """An auxiliary class, used for parsing, that contains the prompt from the
    user.

    Attributes:
        prompt (str): the prompt from the user.
    """
    prompt: str
