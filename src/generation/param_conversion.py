from llm_sdk.llm_sdk import Small_LLM_Model
from src.models import ParamType


def clean_parameter_output(
        generated: list[int],
        param_type: ParamType, model: Small_LLM_Model) -> str:
    """A function that cleans the generated parameter by the LLM model.

    Args:
        generated (list[int]): a list of the generated tokens by the model.
        param_type (ParamType): a class indicating the type of the parameter.
        model (Small_LLM_Model): the LLM model.

    Returns:
        str: the generated and cleaned parameter.
    """

    decoded_text = model.decode(generated)
    match param_type.type:
        case "number":
            clean_output = decoded_text.removesuffix(',')
        case "string":
            splitted = decoded_text.split('"')
            clean_output = splitted[0]
        case "boolean":
            clean_output = decoded_text

    return clean_output


def convert_param_value(
        param_type: ParamType, generated: list[int],
        model: Small_LLM_Model) -> float | str | bool:
    """A function that takes the generated parameter by the LLM model and
    returns it in its correct Python type.

    Args:
        param_type (ParamType): a class indicating the type of the parameter.
        generated (list[int]): a list of the generated tokens by the model.
        model (Small_LLM_Model): the LLM model.

    Returns:
        float | str | bool: the generated parameter in its according Python
            type.
    """

    text = clean_parameter_output(generated, param_type, model)
    return_value: float | str | bool
    match param_type.type:
        case "number":
            return_value = float(text)
        case "string":
            return_value = text
        case "boolean":
            return_value = bool(text)

    return return_value
