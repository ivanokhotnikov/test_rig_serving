# Test rig anomaly detection and condition monitoring for predictive maintenance 

The repository contains the development of applications to detect anomalies in test results (anomaly detection), to monitor the test rig condition (forecasting), and to predict a need for maintenance in advance of a rig failure or excessive deterioration of its performance. The code also serves to provide advanced analytics of test article performance over time or during the test cycle.

# Application serving

Both applications (anomaly detector and forecaster) follow the same cloud architecture to continuously build, deploy and serve the app.

![App serving architecture](https://github.com/ivanokhotnikov/test_rig_serving/blob/master/images/serving.png?raw=True)

# Links

[Anomaly detector](https://34.105.255.15:80/)

[Forecasting](https://35.242.138.174:80/)

# Usage

```
streamlit run anomaly_detection/src/serving.py
streamlit run forecasting/src/serving.py
```