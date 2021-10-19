def check_if_valid_palindrome(string_input: str):
    str_lower = string_input.lower()
    str_lower_list = list(str_lower)
    str_lower_list_reversed = str_lower_list[::-1]
    str_lower_reversed = ''.join(str_lower_list_reversed)

    if str_lower_reversed == str_lower:
        return True
    else:
        return False