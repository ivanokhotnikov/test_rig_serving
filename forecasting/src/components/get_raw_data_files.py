import re


def get_raw_data_files(unit):
    """
    The get_raw_data_files function returns a list of all the raw data files in the RAW_DATA_BUCKET bucket that match the unit number provided. The function accepts one parameter, unit, which is an integer representing the unit number of interest. For example, if you wanted to get all of the raw data files for Unit 1, you would call get_raw_data_files(unit=1). If no matching files were found for that particular unit number, an empty list will be returned.
    
    Args:
        unit: Filter the list of files to only those that match the unit number
    
    Returns:
        A list of the raw data files that are stored in the raw_data bucket
    """
    from components import is_name_valid
    from components.constants import RAW_DATA_BUCKET
    return [
        b.name for b in RAW_DATA_BUCKET.list_blobs()
        if str(unit).zfill(4) in re.split(r'_|-|/', b.name)[0][-4:]
        and is_name_valid(b)
    ]
