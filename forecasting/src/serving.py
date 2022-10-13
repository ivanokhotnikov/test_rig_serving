import streamlit as st

from components import (build_power_features, import_final_features,
                        import_metrics, import_model, import_processed_data,
                        in_data_bucket, plot_correlation_matrix, plot_forecast,
                        predict, read_data_file, read_raw_data,
                        remove_step_zero, upload_data,
                        upload_new_raw_data_file)


def main():
    st.set_page_config(layout='centered',
                       page_title='Forecasting',
                       page_icon='fav.png')
    with st.sidebar:
        st.header('Settings')
        st.subheader('General')
        dev_flag = st.checkbox(
            'DEV',
            value=True,
            help=
            'Process only one feature (load and forecast with the single model)',
        )
        st.subheader('Data')
        read_raw_flag = st.checkbox(
            'Read raw data',
            value=False,
            help=
            'When checked, reads and does initial preprocessing of every raw data file in the data storage. When unchecked, imports the processed data file from the data storage',
        )
        uploaded_file = st.file_uploader(
            'Upload raw data file',
            type=['csv'],
            help=
            'The raw data file with the naming according to the test report convention, the format is HYD******-R1_RAW.csv',
        )
        st.subheader('Visualisation')
        plot_each_unit_flag = st.checkbox(
            'Plot each unit',
            value=False,
            help=
            'The flag to colour or to shade out each unit in the forecast plot',
        )
        averaging_window = int(
            st.number_input(
                'Window size of moving average, seconds',
                value=3600,
                min_value=1,
                max_value=7200,
                step=1,
                help=
                'Select the size of moving average window.\n See https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html for implementation.',
            ))
        st.subheader('Forecast')
        forecasting_window = int(
            st.number_input('Forecast with the last, seconds',
                            value=7200,
                            min_value=1000,
                            max_value=100000,
                            step=1,
                            disabled=True if uploaded_file else False))
    st.title('Forecasting test data')
    st.subheader('Data')
    with st.spinner('Reading data and features'):
        current_processed_data_df = read_raw_data(
        ) if read_raw_flag else import_processed_data()
        final_features = import_final_features()
    tab1, tab2, tab3 = st.tabs(
        ['Features Correlation', 'Processed Data', 'Statistics'])
    with tab1:
        st.subheader('Power features correlation')
        st.plotly_chart(plot_correlation_matrix(current_processed_data_df,
                                                final_features),
                        use_container_width=True)
        st.write(
            'For reference and implementation see: \nhttps://en.wikipedia.org/wiki/Pearson_correlation_coefficient \nhttps://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html'
        )
    with tab2:
        st.subheader('Processed dataframe')
        st.dataframe(current_processed_data_df, use_container_width=True)
    with tab3:
        st.subheader('Descriptive statistics')
        st.write(current_processed_data_df[final_features].describe().T.style.
                 background_gradient(cmap='inferno'))
        st.write(
            'For more details see: \nhttps://test-data-profiling.hydreco.uk/')
    st.subheader('Training')
    tab1, tab2 = st.tabs(['Pipeline', 'Architecture'])
    with tab1:
        st.subheader('Directed acyclic training graph')
        st.image(
            'https://raw.githubusercontent.com/ivanokhotnikov/test_rig_forecast_vertex/master/images/training_dag.png'
        )
        st.write(
            'For more implementation details see: \nhttps://github.com/ivanokhotnikov/test_rig_forecast_vertex'
        )
    with tab2:
        st.subheader('Example of model architecture')
        with st.spinner('Loading model example'):
            forecaster = import_model('DRIVE_POWER.h5')
        forecaster.summary(print_fn=lambda x: st.text(x))
        st.write(
            'For the architecture details see: \nhttps://en.wikipedia.org/wiki/Long_short-term_memory \nhttps://keras.io/api/layers/recurrent_layers/lstm/ \nhttp://www.bioinf.jku.at/publications/older/2604.pdf'
        )
    st.header('Forecast')
    if uploaded_file is not None:
        if not in_data_bucket(uploaded_file.name):
            with st.spinner(
                    f'Uploading {uploaded_file.name}, updating processed data.'
            ):
                upload_new_raw_data_file(uploaded_file)
                current_processed_data_df = read_raw_data()
                upload_data(
                    current_processed_data_df,
                    data_type='processed',
                )
            st.write(
                f'{uploaded_file.name} has been uploaded to the data storage.')
            with st.spinner(f'Processing {uploaded_file.name}'):
                new_data_df = read_data_file(uploaded_file)
                new_data_df = remove_step_zero(new_data_df)
                new_data_df = build_power_features(new_data_df)
        else:
            st.write(f'{uploaded_file.name} is already in the data storage.')
    st.write('Plotting feature forecasts')
    progress_bar = st.progress(0)
    for idx, feature in enumerate(final_features, 1):
        progress_bar.progress(idx / len(final_features))
        with st.spinner(f'Plotting {feature} forecast'):
            st.subheader(f'{feature}')
            st.write('Model\'s forecast')
            scaler = import_model(f'{feature}.joblib')
            forecaster = import_model(f'{feature}.h5')
            forecast = predict(
                new_data_df if (uploaded_file is not None
                                and not in_data_bucket(uploaded_file.name))
                else current_processed_data_df.iloc[-forecasting_window:],
                feature,
                scaler,
                forecaster,
            )
            st.plotly_chart(
                plot_forecast(
                    current_processed_data_df,
                    forecast,
                    feature,
                    new=new_data_df if
                    (uploaded_file is not None
                     and not in_data_bucket(uploaded_file.name)) else None,
                    plot_ma_all=True,
                    rolling_window=averaging_window,
                    plot_each_unit=plot_each_unit_flag,
                ),
                use_container_width=True,
            )
            st.write('Model\'s metrics')
            metrics = import_metrics(feature)
            col1, col2 = st.columns(2)
            col1.metric(label=list(metrics.keys())[0].capitalize().replace(
                '_', ' '),
                        value=f'{list(metrics.values())[0]:.3e}')
            col2.metric(label=list(metrics.keys())[1].capitalize().replace(
                '_', ' '),
                        value=f'{list(metrics.values())[1]:.3e}')
        if dev_flag: break


if __name__ == '__main__':
    main()
