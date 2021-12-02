import streamlit as st

from utils.readers import DataReader, Preprocessor, ModelReader
from utils.plotters import Plotter
from utils.config import FEATURES_NO_TIME, FEATURES_NO_TIME_AND_COMMANDS

uploaded_file = st.file_uploader('Upload your data file', type=['csv'])
if uploaded_file is not None:
    uploaded_df = DataReader.read_newcoming_data(uploaded_file)
    if st.button('Show data'):
        st.write('Uploaded Data', uploaded_df)
        preprocessed_df = Preprocessor.remove_step_zero(uploaded_df)
        st.write('Preprocessed Data', preprocessed_df)
        for feature in FEATURES_NO_TIME:
            st.plotly_chart(
                Plotter.plot_unit_per_test_feature(
                    preprocessed_df,
                    unit=preprocessed_df.iloc[0]['UNIT'],
                    feature=feature,
                    save=False,
                    show=False))
