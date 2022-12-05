import streamlit as st


def is_data_valid(uploaded_file):
    """
    The is_data_valid function checks whether the uploaded file has a valid name and schema. It returns True if both are valid, False otherwise.
    
    Args:
        uploaded_file: Pass the file to the is_name_valid and is_schema_valid functions
    
    Returns:
        True if the uploaded file is valid, False otherwise
    """
    from components import is_name_valid, is_schema_valid
    name_valid = is_name_valid(uploaded_file)
    schema_valid = is_schema_valid(uploaded_file)
    if name_valid:
        st.success('Name OK', icon='✅')
    else:
        st.error('Name NOK', icon='🚨')
    if schema_valid:
        st.success('Schema OK', icon='✅')
    else:
        st.error('Schema NOK', icon='🚨')
    return all([name_valid, schema_valid])
