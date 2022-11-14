# Test rig condition monitoring for predictive maintenance

The repository contains the development of applications to detect anomalies in test results (anomaly detection), to monitor and predict the test rig condition (forecasting). The overall purpose is to implement predictive maintenance and to identify a need for maintenance in advance of a rig failure or excessive deterioration of its performance. The code also serves to provide advanced analytics of test article performance over time or during the test cycle.

## Application serving

Both applications (anomaly detector and forecaster) follow the same cloud architecture to continuously test, build, deploy and serve the app. The apps updates are triggered by the code base modification in the this repository (push-to-master trigger). The build specs can be found in `cloudbuild` folder and include the `pytest` testing, Docker image building and pushing to Google  Container Registry (GCR).

The apps include file upload boxes to enable new data analysis. The file uploader will ingest and validate the new coming raw data file. In case the raw data file is valid and new indeed (no such data file was found in the raw data storage in the raw data storage, Google Cloud Storage (GCS) `test_rig_raw_data` bucket), this file will be uploaded to the raw data storage and will trigger execution of the [training pipeline](https://github.com/ivanokhotnikov/test_rig_forecast_training).

![App serving architecture](https://github.com/ivanokhotnikov/test_rig_serving/blob/master/images/serving.png?raw=True)

## Links

The following links direct to the apps.

[Anomaly detector](http://detector.hydreco.uk/)

[Forecaster](http://forecaster.hydreco.uk/)

## Usage

```python
streamlit run anomaly_detection/src/serving.py
streamlit run forecasting/src/serving.py
```
