def is_name_valid(file):
    """
    The is_name_valid function checks if the file name ends with .csv and contains the word RAW. It returns True if both conditions are met, False otherwise.
    
    Args:
        file: Check if the file name ends with 
    
    Returns:
        A boolean value
    """
    return file.name.endswith('.csv') and ('RAW' in file.name)
