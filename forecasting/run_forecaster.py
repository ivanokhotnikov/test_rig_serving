import streamlit as st
from utils.config import RAW_FORECAST_FEATURES, FORECAST_FEATURES
from utils.readers import get_processed_data
from utils.plotters import plot_heatmap, plot_ma_trend

if __name__ == '__main__':
    df = get_processed_data(raw=False,
                            local=False,
                            features_to_read=RAW_FORECAST_FEATURES)
    if st.button('Plot trends with moving average'):
        window = st.number_input('Window size, seconds',
                                 value=3600,
                                 min_value=1,
                                 max_value=5000,
                                 step=1)
        for feature in FORECAST_FEATURES:
            st.plotly_chart(plot_ma_trend(df, feature, window, show=False))
    if st.button('Plot heatmap of features'):
        st.plotly_chart(plot_heatmap(df, FORECAST_FEATURES, show=False))
