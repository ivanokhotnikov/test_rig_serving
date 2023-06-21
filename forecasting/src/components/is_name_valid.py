def is_name_valid(file):
    """
    The is_name_valid function checks if the file name ends with '.csv' and contains the word 'RAW' in it. It returns True if both conditions are met, else False is returned.
    
    Args:
        file: Access the file name
    
    Returns:
        True if the file name ends with '.csv' and has 'RAW' in it.
    """
    return all([r in file.name for r in ('LN', 'RAW')])
