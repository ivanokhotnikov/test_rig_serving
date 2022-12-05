def remove_step_zero(df):
    """
    The remove_step_zero function removes the rows in a dataframe that have a value of 0 for the 'STEP' column. The function returns a new dataframe with these rows removed and resets the index to avoid any issues that may arise from having non-consecutive indices.
    
    Args:
        df: Specify the dataframe to be used
    
    Returns:
        A pandas dataframe without step zero and reindexed
    """
    return df.drop(df[df['STEP'] == 0].index, axis=0).reset_index(drop=True)
