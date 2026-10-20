from llm_sdk.llm_sdk import Small_LLM_Model
from .masking import (filter_digits, mask_logits,
                      build_forbidden_tokens, allow_all_except)
import numpy as np
from src.models import ParamType


class DataGenerator():

    _model: Small_LLM_Model
    _data: dict[str, int]
    _forbidden_tokens: list[int]

    def __init__(self, model: Small_LLM_Model, data: dict[str, int]):

        self._model = model
        self._data = data
        self._forbidden_tokens = build_forbidden_tokens(model)

    def _select_mask(self, type: str,
                     logits: list[float]) -> tuple[list[float], int]:

        match type:
            case "number":
                stopper = self._data[","]
                mask = filter_digits(self._data)
                mask.append(stopper)
                masked = mask_logits(logits, mask)
            case "string":
                stopper = self._data['"']
                masked = allow_all_except(logits, self._forbidden_tokens)
            case "boolean":
                mask = [self._data["true"], self._data["false"]]
                masked = mask_logits(logits, mask)
                stopper = -1

        return masked, stopper

    def generate_answer(self, alive: list[list[int]],
                        stopper: int,
                        input_ids: list[int]) -> tuple[list
                                                       [int],
                                                       list[int]]:

        ids_copy = list(input_ids)
        step = 0
        generated: list[int] = []
        while True:

            allowed = []
            helper = []
            for candidate in alive:
                if step == len(candidate):
                    allowed.append(stopper)
                elif candidate[step] not in allowed:
                    allowed.append(candidate[step])

            logits = self._model.get_logits_from_input_ids(ids_copy)
            masked = mask_logits(logits, allowed)
            best = int(np.argmax(masked))

            ids_copy.append(best)
            generated.append(best)

            for candidate in alive:
                if step < len(candidate):
                    if candidate[step] == best:
                        helper.append(candidate)

            alive = helper
            step += 1
            if best == stopper:
                break

        return generated, ids_copy

    def generate_parameter(self, param_type: ParamType, input_ids: list[int]
                           ) -> tuple[list[int], list[int]]:

        ids_copy = list(input_ids)
        generated: list[int] = []
        while True:

            logits = self._model.get_logits_from_input_ids(ids_copy)
            masked, stopper = self._select_mask(
                param_type.type, logits)
            best = int(np.argmax(masked))

            ids_copy.append(best)
            generated.append(best)

            if param_type.type == "string":
                decoded_token = self._model.decode([best])
                if '"' in decoded_token:
                    break
            elif param_type.type == "boolean":
                break
            else:
                if best == stopper:
                    break

        return generated, ids_copy
