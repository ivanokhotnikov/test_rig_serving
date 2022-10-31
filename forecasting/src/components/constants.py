import google.cloud.storage as storage

RAW_FORECAST_FEATURES = [
    'TIME', 'STEP', 'HSU DEMAND', 'M1 SPEED', 'M1 CURRENT', 'M1 TORQUE',
    'PT4 SETPOINT', 'PT4', 'D1 RPM', 'D1 CURRENT', 'D1 TORQUE', 'M2 RPM',
    'M2 Amp', 'M2 Torque', 'CHARGE PT', 'CHARGE FLOW', 'M3 RPM', 'M3 Amp',
    'M3 Torque', 'Servo PT', 'SERVO FLOW', 'M4 ANGLE', 'HSU IN', 'TT2',
    'HSU OUT', 'M5 RPM', 'M5 Amp', 'M5 Torque', 'M6 RPM', 'M6 Amp',
    'M6 Torque', 'M7 RPM', 'M7 Amp', 'M7 Torque', 'Vibration 1', ' Vibration 2'
]
COMMANDS = ['TIME', ' DATE', 'STEP', 'HSU DEMAND', 'PT4 SETPOINT']
FEATURES_NO_TIME = [
    f for f in RAW_FORECAST_FEATURES if f not in ('TIME', ' DATE', 'DATE')
]
FEATURES_NO_TIME_AND_COMMANDS = [
    f for f in FEATURES_NO_TIME if f not in COMMANDS
]

ENGINEERED_FEATURES = [
    'DRIVE POWER', 'LOAD POWER', 'CHARGE MECH POWER', 'CHARGE HYD POWER',
    'SERVO MECH POWER', 'SERVO HYD POWER', 'SCAVENGE POWER',
    'MAIN COOLER POWER', 'GEARBOX COOLER POWER', 'UNIT', 'TEST'
]
PRESSURE_TEMPERATURE = ['PT4', 'HSU IN', 'TT2', 'HSU OUT']
VIBRATIONS = ['Vibration 1', ' Vibration 2']

FORECAST_FEATURES = [
    f.strip().replace(' ', '_') for f in ENGINEERED_FEATURES + VIBRATIONS
]
TIME_FEATURES = ['TIME', 'DURATION', 'TOTAL SECONDS', 'RUNNING HOURS']

PROJECT_ID = 'test-rig-349313'
REGION = 'europe-west2'
PROJECT_NUMBER = 42869708044

RAW_DATA_BUCKET_NAME = 'test_rig_raw_data'
RAW_DATA_BUCKET_URI = f'gs://{RAW_DATA_BUCKET_NAME}'
INTERIM_DATA_BUCKET_NAME = 'test_rig_interim_data'
INTERIM_DATA_BUCKET_URI = f'gs://{INTERIM_DATA_BUCKET_NAME}'
PROCESSED_DATA_BUCKET_NAME = 'test_rig_processed_data'
PROCESSED_DATA_BUCKET_URI = f'gs://{PROCESSED_DATA_BUCKET_NAME}'
PIPELINES_BUCKET_NAME = 'test_rig_pipelines'
PIPELINES_BUCKET_URI = f'gs://{PIPELINES_BUCKET_NAME}'
FEATURES_BUCKET_NAME = 'test_rig_features'
FEATURES_BUCKET_URI = f'gs://{FEATURES_BUCKET_NAME}'
MODELS_BUCKET_NAME = 'models_forecasting'
MODELS_BUCKET_URI = f'gs://{MODELS_BUCKET_NAME}'

STORAGE_CLIENT = storage.Client()
RAW_DATA_BUCKET = STORAGE_CLIENT.get_bucket(RAW_DATA_BUCKET_NAME)
INTERIM_DATA_BUCKET = STORAGE_CLIENT.get_bucket(INTERIM_DATA_BUCKET_NAME)
PROCESSED_DATA_BUCKET = STORAGE_CLIENT.get_bucket(PROCESSED_DATA_BUCKET_NAME)
MODELS_BUCKET = STORAGE_CLIENT.get_bucket(MODELS_BUCKET_NAME)
FEATURES_BUCKET = STORAGE_CLIENT.get_bucket(FEATURES_BUCKET_NAME)

LOOKBACK = 120
