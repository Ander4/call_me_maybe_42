def find_closing_quote(text: str) -> int | None:
    """Returns the index of the first unescaped double quote in text.

    A backslash escapes the character that follows it, so \\" is content
    while \\\\" (an escaped backslash followed by a quote) closes the
    value.

    Args:
        text (str): the accumulated text generated for a string value.

    Returns:
        int | None: the index of the closing quote, or None if there isn't
            one.
    """
    i = 0
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == '"':
            return i
        i += 1
    return None
