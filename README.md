# Test rig condition monitoring and anomaly detection for predictive maintenance 

The repository contains the development of an algorithm to monitor the test rig condition, to detect anomalies in test results, and to predict a need for maintenance in advance of a rig failure or excessive deterioration of its performance. The code also serves to provide advanced analytics of test article performance over time or during the test cycle.

## Utils

The directory contains the utility modules to pre- and post-process (see `readers.py`), to visualize (see `plotters.py`) the development results, to save and load preliminary output models, images, etc. The configurations are stored in  the `config.py` governing features sets, hyper-parameters tuning and directories.