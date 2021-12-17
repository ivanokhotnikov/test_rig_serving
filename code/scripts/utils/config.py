import os

PROJECT_DIR = '.'
IMAGES_PATH = os.path.join(PROJECT_DIR, 'outputs', 'images')
MODELS_PATH = os.path.join(PROJECT_DIR, 'outputs', 'models')
PREDICTIONS_PATH = os.path.join(PROJECT_DIR, 'outputs', 'predictions')
DATA_PATH = os.path.join(PROJECT_DIR, 'data')

FEATURES = [
    'TIME', 'STEP', 'HSU DEMAND', 'M1 SPEED', 'M1 CURRENT', 'M1 TORQUE',
    'PT4 SETPOINT', 'PT4', 'D1 RPM', 'D1 CURRENT', 'D1 TORQUE', 'M2 RPM',
    'M2 Amp', 'M2 Torque', 'CHARGE PT', 'CHARGE FLOW', 'M3 RPM', 'M3 Amp',
    'M3 Torque', 'Servo PT', 'SERVO FLOW', 'M4 ANGLE', 'HSU IN', 'TT2',
    'HSU OUT', 'M5 RPM', 'M5 Amp', 'M5 Torque', 'M6 RPM', 'M6 Amp',
    'M6 Torque', 'M7 RPM', 'M7 Amp', 'M7 Torque', 'Vibration 1',
    ' Vibration 2', ' DATE'
]
COMMANDS = ['TIME', ' DATE', 'STEP', 'HSU DEMAND', 'PT4 SETPOINT']
FEATURES_NO_TIME = [f for f in FEATURES if f not in ('TIME', ' DATE')]
FEATURES_NO_TIME_AND_COMMANDS = [
    f for f in FEATURES_NO_TIME if f not in COMMANDS
]
TRIAL_DRIFT = [
    'STEP', 'UNIT', 'TEST', 'M3 Amp', 'M3 Torque', 'Servo PT', 'HSU IN', 'TT2',
    'HSU OUT', 'M7 Amp', 'M7 Torque'
]

FOLDS = 5
SEED = 42
VERBOSITY = 100
EARLY_STOPPING_ROUNDS = 100
OPTIMIZATION_TIME_BUDGET = 5 * 60 * 60
