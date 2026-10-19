from llm_sdk.llm_sdk import Small_LLM_Model
from src.models import Function, Prompt


class PromptGen():

    _functions_info: str
    _model: Small_LLM_Model

    def __init__(self, functions_list: list[Function], model: Small_LLM_Model):
        self._build_functions_info(functions_list)
        self._model = model

    def _build_functions_info(self, functions_list: list[Function]) -> None:

        self._functions_info = ""
        for function in functions_list:
            self._functions_info += f"\n- {
                function.name}: {
                function.description} "

    def build_chat_prompt(
            self, user_prompt: Prompt) -> list[int]:

        formatted_prompt = (
            f"<|im_start|>user\nAvailable functions:{self._functions_info}\n\n"
            f"Question: {user_prompt.prompt}<|im_end|>\n"
            "<|im_start|>assistant\n</think>\n"
        )
        sentence = self._model.encode(formatted_prompt)
        input_ids = sentence[0].tolist()

        return input_ids
