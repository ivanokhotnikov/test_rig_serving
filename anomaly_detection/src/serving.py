import numpy as np
import streamlit as st

from components import (build_power_features, import_model, is_data_valid,
                        create_sequences, is_in_data_bucket, plot_anomalies,
                        plot_unit, read_data_file, read_raw_data,
                        remove_step_zero, upload_new_raw_data_file,
                        upload_processed_data)
from components.constants import LOOKBACK, ENGINEERED_FEATURES

st.set_page_config(
    layout='centered',
    page_title='Detector',
    page_icon=
    'https://github.com/ivanokhotnikov/test_rig_serving/blob/master/images/fav.png?raw=True',
)


def main():
    st.header('Anomaly detection in unit test data')
    uploaded_file = st.file_uploader('Upload raw data file', type=['csv'])
    if uploaded_file is not None:
        st.header('Uploaded raw data')
        with st.spinner(f'Validating the uploaded data file'):
            data_valid = is_data_valid(uploaded_file)
        in_bucket = is_in_data_bucket(uploaded_file)
        st.info(f'{"Not" if not in_bucket else "Already"} in the data storage',
                icon='ℹ️')
        if data_valid:
            if not in_bucket:
                with st.spinner(f'Uploading {uploaded_file.name}'):
                    upload_new_raw_data_file(uploaded_file)
                with st.spinner('Updating processed data'):
                    current_processed_df = read_raw_data()
                    upload_processed_data(current_processed_df)
                st.write(
                    f'{uploaded_file.name} has been uploaded to the data storage.'
                )
            with st.spinner(f'Processing {uploaded_file.name}'):
                new_data_df = read_data_file(uploaded_file)
                new_data_df_wo_zero = remove_step_zero(new_data_df)
                new_interim_df = build_power_features(new_data_df_wo_zero)
            tab1, tab2, tab3 = st.tabs(
                ['Uploaded raw data', 'Uploaded raw data plots', 'Anomalies'])
            with tab1:
                st.subheader('Dataframe of the uploaded unit data')
                st.dataframe(new_data_df, use_container_width=True)
            with tab2:
                st.subheader('Plots of the uploaded unit data')
                with st.spinner('Plotting uploaded unit data'):
                    for feature in new_data_df.columns:
                        if feature not in ('TIME', 'DURATION', 'NOT USED',
                                           ' DATE', 'UNIT', 'TEST'):
                            st.plotly_chart(plot_unit(new_data_df, feature),
                                            use_container_width=True)
            with tab3:
                algorithm = st.radio(
                    'Select algorithm',
                    (
                        None,
                        'IsolationForest',
                        'LocalOutlierFactor',
                        'ConvolutionalAutoencoder',
                    ),
                    help=
                    'References on isolation forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html, local outlier factor: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html, convolutional autoencoder: https://keras.io/examples/timeseries/timeseries_anomaly_detection/',
                    index=0,
                )
                if algorithm is not None:
                    detecting_bar = st.progress(0)
                    for idx, feature in enumerate(ENGINEERED_FEATURES, 1):
                        detecting_bar.progress(idx / len(ENGINEERED_FEATURES))
                        detector = import_model(f'{algorithm}_{feature}')
                        if algorithm == 'ConvolutionalAutoencoder':
                            scaler = import_model(
                                f'{algorithm}_{feature}_scaler')
                            scaled_data = scaler.transform(
                                new_interim_df[feature.replace(
                                    " ", "_")].values.reshape(-1, 1))
                            x = create_sequences(scaled_data,
                                                 lookback=LOOKBACK)
                            pred = detector.predict(x)
                            threshold = import_model(
                                f'{algorithm}_{feature}_threshold')
                            test_mae_loss = np.mean(np.abs(pred - x), axis=1)
                            test_mae_loss = test_mae_loss.reshape((-1))
                            anomalies = test_mae_loss > threshold
                            anomalous_data_indices = []
                            for data_idx in range(
                                    LOOKBACK - 1,
                                    len(scaled_data) - LOOKBACK + 1,
                            ):
                                if np.all(anomalies[data_idx - LOOKBACK +
                                                    1:data_idx]):
                                    anomalous_data_indices.append(data_idx)
                            new_interim_df.loc[:,
                                               f'ANOMALY_{feature.replace(" ","_")}'] = 1
                            new_interim_df.loc[
                                anomalous_data_indices,
                                f'ANOMALY_{feature.replace(" ","_")}'] = -1
                        else:
                            new_interim_df[
                                f'ANOMALY_{feature.replace(" ","_")}'] = detector.predict(
                                    new_interim_df[feature.replace(
                                        " ", "_")].values.reshape(-1, 1))
                        try:
                            st.plotly_chart(
                                plot_anomalies(
                                    new_interim_df,
                                    unit=new_interim_df.iloc[0]['UNIT'],
                                    test=1,
                                    feature=feature))
                        except:
                            st.write(
                                f'No anomalies found in unit {new_interim_df.iloc[0]["UNIT"]} for {feature} with {algorithm}'
                            )
                            continue


if __name__ == '__main__':
    main()
