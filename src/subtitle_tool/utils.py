def sanitize_int(number: int | None, unknown_to_zero: bool = True) -> int:
    """
    This function ensures that Nones are treated
    as 0 during execution, ensuring that int operations
    can be performed over the values.

    Args:
        number (int | None): number to check
        unknown_zero (bool): whether to force anything unknown
            to become a 0. (default: True)

    Returns:
        int: number passed or 0 if None
    """
    # Covering nones
    if not number:
        return 0

    if isinstance(number, int):
        return number

    # Covering types that are not ints but are also numbers
    if isinstance(number, float):
        return int(number)

    # Covering all other types
    if not unknown_to_zero:
        raise ValueError(f"{number} is not an int or a number")

    # If we got here this means we should return 0 for unknowns or
    # invalids.
    return 0
