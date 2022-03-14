import os

PROJECT_DIR = '.'
DATA_PATH = os.path.join(PROJECT_DIR, 'data')
IMAGES_PATH = os.path.join(PROJECT_DIR, 'outputs', 'images')
MODELS_PATH = os.path.join(PROJECT_DIR, 'outputs', 'models')
PREDICTIONS_PATH = os.path.join(PROJECT_DIR, 'outputs', 'predictions')

FEATURES_FOR_FORECASTING = [
    'TIME', 'STEP', 'HSU DEMAND', 'M1 SPEED', 'M1 CURRENT', 'M1 TORQUE',
    'PT4 SETPOINT', 'PT4', 'D1 RPM', 'D1 CURRENT', 'D1 TORQUE', 'M2 RPM',
    'M2 Amp', 'M2 Torque', 'CHARGE PT', 'CHARGE FLOW', 'M3 RPM', 'M3 Amp',
    'M3 Torque', 'Servo PT', 'SERVO FLOW', 'M4 ANGLE', 'HSU IN', 'TT2',
    'HSU OUT', 'M5 RPM', 'M5 Amp', 'M5 Torque', 'M6 RPM', 'M6 Amp',
    'M6 Torque', 'M7 RPM', 'M7 Amp', 'M7 Torque', 'Vibration 1',
    ' Vibration 2', ' DATE'
]

FEATURES_FOR_ANOMALY_DETECTION = [
    'TIME', 'STEP', 'HSU DEMAND', 'M1 SPEED', 'M1 CURRENT', 'M1 TORQUE',
    'PT4 SETPOINT', 'PT4', 'D1 RPM', 'D1 CURRENT', 'D1 TORQUE', 'M2 RPM',
    'M2 Amp', 'M2 Torque', 'CHARGE PT', 'CHARGE FLOW', 'M3 RPM', 'M3 Amp',
    'M3 Torque', 'Servo PT', 'SERVO FLOW', 'M4 ANGLE', 'HSU IN', 'TT2',
    'HSU OUT', 'M5 RPM', 'M5 Amp', 'M5 Torque', 'M6 RPM', 'M6 Amp',
    'M6 Torque', 'M7 RPM', 'M7 Amp', 'M7 Torque'
]

ENGINEERED_FEATURES = [
    'DRIVE POWER', 'LOAD POWER', 'CHARGE MECH POWER', 'CHARGE HYD POWER',
    'SERVO MECH POWER', 'SERVO HYD POWER', 'SCAVENGE POWER',
    'MAIN COOLER POWER', 'GEARBOX COOLER POWER'
]
PRESSURE_TEMPERATURE_FEATURES = ['PT4', 'HSU IN', 'TT2', 'HSU OUT']
COMMANDS = ['TIME', ' DATE', 'STEP', 'HSU DEMAND', 'PT4 SETPOINT']
FEATURES_NO_TIME = [
    f for f in FEATURES_FOR_ANOMALY_DETECTION if f not in ('TIME', ' DATE')
]
FEATURES_NO_TIME_AND_COMMANDS = [
    f for f in FEATURES_NO_TIME if f not in COMMANDS
]

FOLDS = 5
SEED = 42
VERBOSITY = 1
OPTIMIZATION_TIME_BUDGET = 5 * 60 * 60
TIME_STEPS = 120
EARLY_STOPPING = 5
