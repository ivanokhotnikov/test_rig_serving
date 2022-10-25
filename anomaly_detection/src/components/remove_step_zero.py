def remove_step_zero(df):
    return df.drop(
        df[df['STEP'] == 0].index,
        axis=0,
    ).reset_index(drop=True)