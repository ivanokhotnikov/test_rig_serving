import streamlit as st

from components import (build_power_features, import_processed_features,
                        import_metrics, import_model, import_processed_data,
                        is_in_data_bucket, is_data_valid,
                        plot_correlation_matrix, plot_forecast, plot_unit,
                        predict, read_data_file, read_latest_unit,
                        read_raw_data, remove_step_zero,
                        upload_new_raw_data_file, upload_processed_data)


def main():
    st.set_page_config(
        layout='centered',
        page_title='Forecasting',
        page_icon=
        'https://github.com/ivanokhotnikov/test_rig_serving/blob/master/images/fav.png?raw=True',
        initial_sidebar_state='collapsed',
    )
    st.title('Forecasting test data')
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
            'The raw data file with the naming according to the test report convention, the format is HYD000XXX-R1_RAW.csv, where XXX - unit number',
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
            tab1, tab2 = st.tabs(
                ['Uploaded raw data', 'Uploaded raw data plots'])
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
    st.header('Data overview')
    with st.spinner('Reading data and features'):
        current_processed_df = read_raw_data(
        ) if read_raw_flag else import_processed_data()
        proceseed_features = import_processed_features()
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        'Features Correlation', 'Raw Data', 'Raw Data Plots', 'Processed Data',
        'Statistics'
    ])
    with tab1:
        st.subheader('Power features correlation')
        with st.spinner('Plotting correlation matrix'):
            st.plotly_chart(plot_correlation_matrix(current_processed_df,
                                                    proceseed_features),
                            use_container_width=True)
            st.write(
                'For reference and implementation see: \nhttps://en.wikipedia.org/wiki/Pearson_correlation_coefficient \nhttps://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html'
            )
    with tab2:
        st.subheader('Dataframe of the latest valid raw unit data')
        with st.spinner('Reading latest raw data file'):
            latest_unit_df = read_latest_unit(current_processed_df)
            st.dataframe(latest_unit_df, use_container_width=True)
    with tab3:
        st.subheader('Plots of the latest valid raw unit data')
        with st.spinner('Plotting latest raw data file'):
            for feature in latest_unit_df.columns:
                if feature not in ('TIME', 'DURATION', 'NOT USED', ' DATE'):
                    st.plotly_chart(plot_unit(latest_unit_df, feature),
                                    use_container_width=True)
    with tab4:
        st.subheader('Processed dataframe')
        st.dataframe(current_processed_df, use_container_width=True)
    with tab5:
        st.subheader('Descriptive statistics')
        st.write(current_processed_df[proceseed_features].describe().T.style.
                 background_gradient(cmap='inferno'))
        st.write(
            'For more details see: \nhttps://test-data-profiling.hydreco.uk/')
    st.header('Training overview')
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
    st.write('Plotting feature forecasts')
    progress_bar = st.progress(0)
    for idx, feature in enumerate(proceseed_features, 1):
        progress_bar.progress(idx / len(proceseed_features))
        with st.spinner(f'Plotting {feature} forecast'):
            st.subheader(f'{feature.lower().capitalize().replace("_", " ")}')
            st.write('Model\'s forecast')
            scaler = import_model(f'{feature}.joblib')
            forecaster = import_model(f'{feature}.h5')
            forecast = predict(
                new_interim_df
                if uploaded_file is not None and not in_bucket and data_valid
                else current_processed_df.iloc[-forecasting_window:],
                feature,
                scaler,
                forecaster,
            )
            st.plotly_chart(
                plot_forecast(
                    current_processed_df,
                    forecast,
                    feature,
                    new=new_interim_df if uploaded_file is not None
                    and not in_bucket and data_valid else None,
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
                        value=f'{list(metrics.values())[0]:.2e}')
            col2.metric(label=list(metrics.keys())[1].capitalize().replace(
                '_', ' '),
                        value=f'{list(metrics.values())[1]:.2e}')
            st.write('Trained on:')
            st.write(
                f'{list(metrics.keys())[2].capitalize().replace("_", " ")} {list(metrics.values())[2]}'
            )
        if dev_flag: break


if __name__ == '__main__':
    main()
