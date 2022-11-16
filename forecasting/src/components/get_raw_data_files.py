import re

from streamlit import cache


@cache(allow_output_mutation=True)
def get_raw_data_files(unit):
    from components import is_name_valid
    from components.constants import RAW_DATA_BUCKET
    return [
        b.name for b in RAW_DATA_BUCKET.list_blobs()
        if str(unit).zfill(4) in re.split(r'_|-|/', b.name)[0][-4:]
        and is_name_valid(b)
    ]
