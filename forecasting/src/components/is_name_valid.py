def is_name_valid(file):
    return file.name.endswith('.csv') and ('RAW' in file.name[3:])
