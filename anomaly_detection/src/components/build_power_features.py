import math


def build_power_features(df):
    df['DRIVE_POWER'] = (df['M1 SPEED'] * df['M1 TORQUE'] * math.pi / 30 /
                         1e3).astype(float)
    df['LOAD_POWER'] = abs(df['D1 RPM'] * df['D1 TORQUE'] * math.pi / 30 /
                           1e3).astype(float)
    df['CHARGE_MECH_POWER'] = (df['M2 RPM'] * df['M2 Torque'] * math.pi / 30 /
                               1e3).astype(float)
    df['CHARGE_HYD_POWER'] = (df['CHARGE PT'] * 1e5 * df['CHARGE FLOW'] *
                              1e-3 / 60 / 1e3).astype(float)
    df['SERVO_MECH_POWER'] = (df['M3 RPM'] * df['M3 Torque'] * math.pi / 30 /
                              1e3).astype(float)
    df['SERVO_HYD_POWER'] = (df['Servo PT'] * 1e5 * df['SERVO FLOW'] * 1e-3 /
                             60 / 1e3).astype(float)
    df['SCAVENGE_POWER'] = (df['M5 RPM'] * df['M5 Torque'] * math.pi / 30 /
                            1e3).astype(float)
    df['MAIN_COOLER_POWER'] = (df['M6 RPM'] * df['M6 Torque'] * math.pi / 30 /
                               1e3).astype(float)
    df['GEARBOX_COOLER_POWER'] = (df['M7 RPM'] * df['M7 Torque'] * math.pi /
                                  30 / 1e3).astype(float)
    return df
