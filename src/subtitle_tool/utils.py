def sanitize_int(number: int | None) -> int:
    """
    This function ensures that Nones are treated
    as 0 during execution, ensuring that int operations
    can be performed over the values.

    Args:
        number (int | None): number to check

    Returns:
        int: number passed or 0 if None
    """
    if not number:
        return 0

    if not isinstance(number, int):
        raise ValueError(f"{number} is not an int")

    return number
