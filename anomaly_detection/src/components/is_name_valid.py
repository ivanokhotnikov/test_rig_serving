def is_name_valid(uploaded_file):
    return uploaded_file.name.endswith('.csv') and ('RAW'
                                                    in uploaded_file.name[3:])
