from llm_sdk.llm_sdk import Small_LLM_Model
from src.models import Function, Prompt


class PromptGen():
    """Class that generates the prompt that will be given to the model.

    Attributes:
            functions_info (str): the information of the available functions.
            model (Small_LLM_Model): the LLM model.
    """

    _functions_info: str
    _model: Small_LLM_Model

    def __init__(self, functions_list: list[Function], model: Small_LLM_Model):
        self._build_functions_info(functions_list)
        self._model = model

    def _build_functions_info(self, functions_list: list[Function]) -> None:
        """A function that builds the functions info from a list of Functions.

        Args:
            functions_list (list[Function]): the list of functions
        """
        self._functions_info = ""
        for function in functions_list:
            self._functions_info += f"\n- {
                function.name}: {
                function.description} "

    def _build_chat_prompt(
            self, user_prompt: Prompt) -> str:
        """A function that builds the prompt from a given user prompt.
        The built prompt will be given to the model in a format that it
        understands.

        Args:
            user_promt (Prompt): the prompt from the user.

        Returns:
            str: the prompt in a format that the model understands.
        """

        formatted_prompt = (
            f"<|im_start|>user\nAvailable functions:{self._functions_info}\n\n"
            f"Question: {user_prompt.prompt}<|im_end|>\n"
            "<|im_start|>assistant\n</think>\n"
        )

        return formatted_prompt

    def get_input_ids(self, user_prompt: Prompt) -> list[int]:
        """A function that returns the input ids generated from the prompt.

        Args:
            user_promt (Prompt): the prompt from the user.

        Returns:
            list[int]: the list of the generated input_ids from the prompt.
        """
        prompt = self._build_chat_prompt(user_prompt)
        sentence = self._model.encode(prompt)
        input_ids = sentence[0].tolist()

        return input_ids
