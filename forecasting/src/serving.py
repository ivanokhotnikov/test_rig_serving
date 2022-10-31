import argparse

import streamlit as st

from components import (build_power_features, get_raw_data_folder_stats,
                        import_forecast_features, import_metrics, import_model,
                        is_data_valid, is_in_data_bucket,
                        plot_correlation_matrix, plot_forecast, plot_unit,
                        predict, read_latest_unit, read_processed_data,
                        read_raw_data, read_unit_data, remove_step_zero,
                        upload_new_raw_data_file, upload_processed_data)


def main(prod_flag_value, plot_each_unit_flag_value, plot_ma_flag_value):
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
        prod_flag = st.checkbox(
            'Process all features',
            value=prod_flag_value,
            help='Process all features (load and forecast)',
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
            value=plot_each_unit_flag_value,
            help=
            'The flag to colour or to shade out each unit in the forecast plot',
        )
        rolling_window_flag = st.checkbox(
            'Plot moving avearge',
            value=plot_ma_flag_value,
            help='The flag to cbuild and plot moving average',
        )
        rolling_window = None
        if rolling_window_flag:
            rolling_window = st.number_input(
                'Window size of moving average, seconds',
                value=3600,
                min_value=1,
                max_value=7200,
                step=1,
                help=
                'Select the size of moving average window.\n See https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html for implementation.',
            )
    current_processed_df = None
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
                    current_processed_df = read_processed_data()
                    upload_new_raw_data_file(uploaded_file)
                with st.spinner('Updating processed data'):
                    new_processed_df = read_raw_data()
                    upload_processed_data(new_processed_df)
                st.write(
                    f'{uploaded_file.name} has been uploaded to the data storage.'
                )
            with st.spinner(f'Processing {uploaded_file.name}'):
                new_data_df = read_unit_data(uploaded_file.name)
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
                            st.plotly_chart(
                                plot_unit(new_data_df, feature),
                                use_container_width=True,
                            )
    st.header('Data overview')
    if current_processed_df is None:
        current_processed_df = read_raw_data(
        ) if read_raw_flag else read_processed_data()
    forecast_features = import_forecast_features()
    tab1, tab2, tab3, tab4 = st.tabs(
        ['Features Correlation', 'Raw Data', 'Processed Data', 'Statistics'])
    with tab1:
        st.subheader('Power features correlation')
        with st.spinner('Plotting correlation matrix'):
            st.plotly_chart(
                plot_correlation_matrix(current_processed_df,
                                        forecast_features),
                use_container_width=True,
            )
            st.write(
                'For reference and implementation see: \nhttps://en.wikipedia.org/wiki/Pearson_correlation_coefficient \nhttps://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html'
            )
    with tab2:
        st.subheader('Raw unit data storage')
        num_files, num_valid_files = get_raw_data_folder_stats()
        col1, col2 = st.columns(2)
        col1.metric(label='Number of raw files', value=num_files)
        col2.metric(label='Number of raw files with valid names',
                    value=num_valid_files)
        unit = None
        unit = st.selectbox(
            'Select the unit number to display',
            current_processed_df['UNIT'].unique(),
            index=len(current_processed_df['UNIT'].unique()) - 1,
        )
        tab21, tab22 = st.tabs(['Dataframe', 'Plots'])
        with tab21:
            st.subheader('Dataframe of the raw unit data')
            with st.spinner('Reading raw data file'):
                unit_df = read_unit_data(
                    f'HYD000{str(unit).zfill(3)}-R1_RAW.csv')
            st.dataframe(unit_df, use_container_width=True)
        with tab22:
            st.subheader('Plots of the raw unit data')
            with st.spinner('Plotting the raw data'):
                for feature in unit_df.columns:
                    if feature not in ('TIME', 'DURATION', 'NOT USED',
                                       ' DATE'):
                        st.plotly_chart(plot_unit(unit_df, feature),
                                        use_container_width=True)
    with tab3:
        st.subheader('Processed dataframe')
        st.dataframe(current_processed_df, use_container_width=True)
    with tab4:
        st.subheader('Descriptive statistics')
        st.dataframe(current_processed_df[forecast_features].describe().T)
        st.write(
            'For more details see: \nhttps://test-data-profiling.hydreco.uk/')
    st.header('Training overview')
    tab1, tab2, tab3, tab4 = st.tabs([
        'Training pipeline', 'Training DAG', 'Model architecture',
        'Serving pipeline'
    ])
    with tab1:
        st.subheader('Training pipeline')
        st.image(
            'https://raw.githubusercontent.com/ivanokhotnikov/test_rig_forecast_training/master/images/training_pipeline.png'
        )
    with tab2:
        st.subheader('Directed acyclic training graph')
        st.image(
            'https://raw.githubusercontent.com/ivanokhotnikov/test_rig_forecast_training/master/images/training_dag.png'
        )
        st.write(
            'For more implementation details see: \nhttps://github.com/ivanokhotnikov/test_rig_forecast_training'
        )
    with tab3:
        st.subheader('Example of model architecture')
        with st.spinner('Loading model example'):
            forecaster = import_model('DRIVE_POWER.h5')
        forecaster.summary(print_fn=lambda x: st.text(x))
        st.write(
            'For the architecture details see: \nhttps://en.wikipedia.org/wiki/Long_short-term_memory \nhttps://keras.io/api/layers/recurrent_layers/lstm/ \nhttp://www.bioinf.jku.at/publications/older/2604.pdf'
        )
    with tab4:
        st.subheader('Serving pipeline')
        st.image(
            'https://raw.githubusercontent.com/ivanokhotnikov/test_rig_serving/master/images/serving.png'
        )
    st.header('Forecast')
    st.write('Plotting feature forecasts')
    progress_bar = st.progress(0)
    for idx, feature in enumerate(forecast_features, 1):
        progress_bar.progress(idx / len(forecast_features))
        with st.spinner(f'Plotting {feature} forecast'):
            st.subheader(f'{feature.lower().capitalize().replace("_", " ")}')
            st.write('Model\'s forecast')
            scaler = import_model(f'{feature}.joblib')
            forecaster = import_model(f'{feature}.h5')
            if uploaded_file is not None and not in_bucket and data_valid:
                new_data = new_interim_df
                new_forecast = predict(
                    data_df=new_data,
                    feature=feature,
                    scaler=scaler,
                    forecaster=forecaster,
                )
                forecast = predict(
                    data_df=current_processed_df[-len(new_data):],
                    feature=feature,
                    scaler=scaler,
                    forecaster=forecaster,
                )
            else:
                new_data = None
                new_forecast = None
                latest_unit_df = read_latest_unit(current_processed_df)
                forecast = predict(
                    data_df=current_processed_df[-len(latest_unit_df):],
                    feature=feature,
                    scaler=scaler,
                    forecaster=forecaster,
                )
            st.plotly_chart(
                plot_forecast(
                    history_df=current_processed_df,
                    forecast=forecast,
                    feature=feature,
                    new_forecast=new_forecast,
                    new_data_df=new_data,
                    rolling_window=rolling_window,
                    plot_each_unit=plot_each_unit_flag,
                ),
                use_container_width=True,
            )
            st.write('Model\'s metrics at training')
            metrics = import_metrics(feature)
            col1, col2 = st.columns(2)
            col1.metric(label=list(metrics.keys())[0].capitalize().replace(
                '_', ' '),
                        value=f'{list(metrics.values())[0]:.2e}')
            col2.metric(label=list(metrics.keys())[1].capitalize().replace(
                '_', ' '),
                        value=f'{list(metrics.values())[1]:.2e}')
            st.write('Model trained on:')
            st.write(
                f'{list(metrics.keys())[2].capitalize().replace("_", " ")} {list(metrics.values())[2]}'
            )
        if not prod_flag: break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prod',
                        help='Production flag (processes all features if set)',
                        action='store_true')
    parser.add_argument('--plot_each_unit',
                        help='Plot each unit flag',
                        action='store_true')
    parser.add_argument('--plot_ma',
                        help='Plot moving average flag',
                        action='store_true')
    args = parser.parse_args()
    main(prod_flag_value=args.prod,
         plot_each_unit_flag_value=args.plot_each_unit,
         plot_ma_flag_value=args.plot_ma)
