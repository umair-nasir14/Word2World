def pad_rows_to_max_length(text):
    """
    Pads each row in the provided text to make them of the length of the row with maximum length.
    Rows are padded with the last character found in that row.
    """
    # Split the text into lines
    lines = text.strip().split("\n")
    
    # Determine the maximum line length
    max_length = max(len(line) for line in lines)
    
    # Pad each line to the maximum length
    padded_lines = [line + line[-1] * (max_length - len(line)) if line else "" for line in lines]
    
    return "\n".join(padded_lines)

def remove_extra_special_chars(input_string):
    """
    This function removes all special characters from the string except one instance of each.
    It does not consider alphabets and numbers as special characters.
    """
    # Finding all special characters (non-alphanumeric and non-whitespace)
    special_chars = set(char for char in input_string if not char.isalnum() and not char.isspace())

    # Create a dictionary to keep track of the first occurrence of each special character
    first_occurrences = {char: False for char in special_chars}

    # Process the string, keeping the first occurrence of each special character
    new_string = []
    for char in input_string:
        if char in special_chars:
            if not first_occurrences[char]:
                new_string.append(char)
                first_occurrences[char] = True
        else:
            new_string.append(char)

    return ''.join(new_string)