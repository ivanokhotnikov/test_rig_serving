import streamlit as st
from components import (
    build_power_features, get_raw_data_files, get_raw_data_folder_stats,
    import_forecast_features, import_forecaster, import_metrics, import_scaler,
    is_data_valid, is_in_data_bucket, plot_correlation_matrix,
    plot_data_distributions, plot_forecast, plot_unit, predict,
    read_latest_unit, read_processed_data, read_raw_data, read_unit_data,
    remove_step_zero, upload_new_raw_data_file, upload_processed_data)


def main():
    st.set_page_config(
        layout='centered',
        page_title='Forecaster',
        page_icon=
        'https://github.com/ivanokhotnikov/test_rig_serving/blob/master/images/fav.png?raw=True',
        initial_sidebar_state='auto')
    st.title('Forecasting test rig data')
    with st.sidebar:
        st.title('Settings')
        st.header('Data')
        uploaded_file = st.file_uploader(
            'Upload raw data file',
            type=['csv'],
            help=
            'The raw data file with the naming according to the test report convention, the format is HYD000XXX-R1_RAW.csv, where XXX - unit number'
        )
        explore_data_flag = st.checkbox('Explore data',
                                        value=False,
                                        help='Show data details',
                                        key='explore_data_flag')
        if explore_data_flag:
            read_raw_flag = st.checkbox(
                'Read raw data',
                value=False,
                help=
                'When checked, reads and does initial preprocessing of every raw data file in the data storage. When unchecked, imports the processed data file from the data storage',
                key='read_raw_flag')
        st.header('Forecast')
        plot_forecast_flag = st.checkbox('Plot forecasts',
                                         value=False,
                                         help='The flag to plot forecast',
                                         key='plot_forecast_flag')
        if plot_forecast_flag:
            plot_each_unit_flag = st.checkbox(
                'Plot each unit',
                value=False,
                help=
                'The flag to colour or to shade out each unit in the forecast plot',
                key='plot_each_unit_flag')
            rolling_window_flag = st.checkbox(
                'Plot moving average',
                value=False,
                help='The flag to build and plot moving average',
                key='rolling_window_flag')
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
                    key='rolling_window')
    st.header('Architecture')
    tab1, tab2, tab3 = st.tabs(
        ['Serving pipeline', 'Training pipeline', 'Training DAG'])
    with tab1:
        st.image(
            'https://raw.githubusercontent.com/ivanokhotnikov/test_rig_serving/master/images/serving.png'
        )
    with tab2:
        st.image(
            'https://raw.githubusercontent.com/ivanokhotnikov/test_rig_forecast_training/master/images/training_pipeline.png'
        )
    with tab3:
        st.image(
            'https://raw.githubusercontent.com/ivanokhotnikov/test_rig_forecast_training/master/images/training_dag.png'
        )
        st.write(
            'For more implementation details see: \nhttps://github.com/ivanokhotnikov/test_rig_forecast_training'
        )
    current_processed_df = None
    forecast_features = None
    if uploaded_file is not None:
        st.header(f'Uploaded raw data file {uploaded_file.name}')
        with st.spinner(f'Validating the uploaded data file'):
            data_valid = is_data_valid(uploaded_file)
        in_bucket = is_in_data_bucket(uploaded_file)
        if data_valid:
            if not in_bucket:
                st.session_state.plot_forecast_flag = True
                st.session_state.plot_each_unit_flag = True
                st.session_state.rolling_window_flag = True
                st.session_state.rolling_window = 3600
                with st.spinner(f'Uploading {uploaded_file.name}'):
                    upload_new_raw_data_file(uploaded_file)
                with st.spinner('Updating processed data'):
                    current_processed_df = read_raw_data()
                    upload_processed_data(current_processed_df)
                st.write(
                    f'{uploaded_file.name} has been uploaded to the raw data storage'
                )
            with st.spinner(f'Processing {uploaded_file.name}'):
                new_data_df = read_unit_data(uploaded_file.name)
                new_data_df_wo_zero = remove_step_zero(new_data_df)
                new_interim_df = build_power_features(new_data_df_wo_zero)
            tab1, tab2 = st.tabs(['Raw data', 'Plots'])
            with tab1:
                st.dataframe(new_data_df, use_container_width=True)
            with tab2:
                for feature in new_data_df.columns:
                    if feature not in ('TIME', 'DURATION', 'NOT USED', ' DATE',
                                       'UNIT', 'TEST'):
                        st.plotly_chart(plot_unit(new_data_df, feature),
                                        use_container_width=True)
    if explore_data_flag:
        st.header('Raw data')
        if current_processed_df is None:
            current_processed_df = read_raw_data(
            ) if read_raw_flag else read_processed_data()
        if forecast_features is None:
            forecast_features = import_forecast_features()
        # tab1, tab2, tab3, tab4 = st.tabs(
        #     ['Raw', 'Processed', 'Statistics', 'Correlations'])
        # with tab1:
        st.subheader('Raw unit data storage')
        num_files, num_valid_files = get_raw_data_folder_stats()
        col1, col2 = st.columns(2)
        col1.metric(label='Number of raw files', value=num_files)
        col2.metric(label='Number of raw files with valid names',
                    value=num_valid_files)
        unit = None
        unit = st.selectbox(
            'Select the unit number to display',
            current_processed_df['UNIT'].unique().astype(int),
            index=len(current_processed_df['UNIT'].unique()) - 1)
        unit_files_list = get_raw_data_files(unit)
        if unit is not None:
            unit_file_name = st.selectbox('Select the data file',
                                            unit_files_list,
                                            index=0)
        tab11, tab12 = st.tabs(['Dataframe', 'Plots'])
        with tab11:
            st.subheader('Dataframe of the unit data')
            unit_df = read_unit_data(unit_file_name)
            st.write(unit_file_name)
            st.dataframe(unit_df, use_container_width=True)
        with tab12:
            st.subheader('Plots of the raw unit data')
            with st.spinner('Plotting the raw data'):
                st.write(unit_file_name)
                for feature in unit_df.columns:
                    if feature not in ('TIME', 'DURATION', 'NOT USED',
                                        'NOT_USED', 'UNIT', 'TEST',
                                        'RUNNING_SECONDS', 'RUNNING_HOURS',
                                        ' DATE', 'DATE'):
                        st.plotly_chart(plot_unit(unit_df, feature),
                                        use_container_width=True)
        st.write('For more details on the processed data see: http://data-profiler.hydreco.uk/')
        # with tab2:
        #     st.dataframe(current_processed_df, use_container_width=True)
        # with tab3:
        #     st.dataframe(current_processed_df[forecast_features].describe().T)
        #     st.write('For more details see: http://data-profiler.hydreco.uk/')
        # with tab4:
        #     st.plotly_chart(plot_correlation_matrix(current_processed_df,
        #                                             forecast_features),
        #                     use_container_width=True)
        #     st.write(
        #         'For reference and implementation see: \nhttps://en.wikipedia.org/wiki/Pearson_correlation_coefficient \nhttps://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html'
        #     )
    if plot_forecast_flag:
        st.header('Forecast')
        if current_processed_df is None:
            current_processed_df = read_processed_data()
        if forecast_features is None:
            forecast_features = import_forecast_features()
        feature = st.selectbox('Select the feature',
                               forecast_features,
                               index=0,
                               key='feature')
        scaler = import_scaler(feature)
        forecaster = import_forecaster(feature)
        if uploaded_file is not None and not in_bucket and data_valid:
            new_data = new_interim_df
            new_forecast = predict(data_df=new_data,
                                   feature=feature,
                                   scaler=scaler,
                                   forecaster=forecaster)
            forecast = predict(data_df=current_processed_df[-len(new_data):],
                               feature=feature,
                               scaler=scaler,
                               forecaster=forecaster)
        else:
            new_data = None
            new_forecast = None
            latest_unit_df = read_latest_unit(current_processed_df)
            forecast = predict(
                data_df=current_processed_df[-len(latest_unit_df):],
                feature=feature,
                scaler=scaler,
                forecaster=forecaster)
        tab1, tab2, tab3, tab4 = st.tabs(
            ['Forecast', 'Skew', 'Model', 'Metrics'])
        with tab1:
            st.plotly_chart(plot_forecast(history_df=current_processed_df,
                                          forecast=forecast,
                                          feature=feature,
                                          new_forecast=new_forecast,
                                          new_data_df=new_data,
                                          rolling_window=rolling_window,
                                          plot_each_unit=plot_each_unit_flag),
                            use_container_width=True)
        with tab2:
            st.plotly_chart(plot_data_distributions(feature,
                                                    current_processed_df,
                                                    forecast=forecast,
                                                    new_data_df=new_data,
                                                    new_forecast=new_forecast),
                            use_container_width=True)
        with tab3:
            forecaster.summary(print_fn=lambda x: st.text(x))
            st.write(
                'For the architecture details see: \nhttps://en.wikipedia.org/wiki/Long_short-term_memory \nhttps://keras.io/api/layers/recurrent_layers/lstm/ \nhttp://www.bioinf.jku.at/publications/older/2604.pdf'
            )
        with tab4:
            metrics = import_metrics(feature)
            col1, col2 = st.columns(2)
            col1.metric(label=list(metrics.keys())[0].capitalize().replace(
                '_', ' '),
                        value=f'{list(metrics.values())[0]:.2e}')
            col2.metric(label=list(metrics.keys())[1].capitalize().replace(
                '_', ' '),
                        value=f'{list(metrics.values())[1]:.2e}')
            st.write(
                f'{list(metrics.keys())[2].capitalize().replace("_", " ")} {list(metrics.values())[2]}'
            )


if __name__ == '__main__':
    main()
