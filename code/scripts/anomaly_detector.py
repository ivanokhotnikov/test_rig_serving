import streamlit as st

from utils.readers import DataReader, Preprocessor, ModelReader
from utils.plotters import Plotter
from utils.config import FEATURES_NO_TIME, FEATURES_NO_TIME_AND_COMMANDS

uploaded_file = st.file_uploader('Upload your data file', type=['csv'])
if uploaded_file is not None:
    uploaded_df = DataReader.read_newcoming_data(uploaded_file)
    df = Preprocessor.remove_step_zero(uploaded_df)
    if st.button('Show data'):
        st.write('Uploaded Data', uploaded_df)
        st.write('Preprocessed Data', df)
    if st.button('Plot data'):
        for feature in FEATURES_NO_TIME:
            st.plotly_chart(
                Plotter.plot_unit_per_test_feature(df,
                                                   unit=df.iloc[0]['UNIT'],
                                                   feature=feature,
                                                   save=False,
                                                   show=False))
if uploaded_file is not None:
    if st.button('Predict anomaly'):
        for feature in FEATURES_NO_TIME_AND_COMMANDS:
            detector = ModelReader.read_model(f'IsolationForest_{feature}')
            df[f'ANOMALY_{feature}'] = detector.predict(
                df[feature].values.reshape(-1, 1))
            try:
                st.plotly_chart(
                    Plotter.plot_anomalies_per_unit_test_feature(
                        df,
                        unit=df.iloc[0]['UNIT'],
                        test=1,
                        feature=feature,
                        show=False))
            except:
                st.write(
                    f'No anomalies found in {df.iloc[0]["UNIT"]} for {feature}'
                )
                continue