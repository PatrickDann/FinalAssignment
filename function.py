def merge_group_avg(data1, data2, grouping_col, avg_col):
    import pandas as pd
    
    """
    Combines two dataframes and groups and averages by a designated columns.
    
    Parameters
    ----------
    data1 : pandas.core.frame.DataFrame
        The first dataframe to combine
    data2 : pandas.core.frame.DataFrame
        The second dataframe to combine
    grouping_col : str
        The column to group the data on
    sum_col : str
        The column to average after grouping
        
    Returns
    -------
    pandas.core.frame.DataFrame 
        A dataframe with the group by column and the result of the action applied.
        
    Raises
    ------
    TypeError
        If the input argument data is not of type pandas.core.frame.DataFrame
    AssertError
        If the input argument grouping_col is not in the data columns
  
    
    Examples
    --------
    >>> merge_group_avg(helper_data1, helper_data2, 'colour', 'ripeness')
    	colour	ripeness
    0	red	    5.0
    1	yellow	4.5
    2	orange	3.0

    """
    
    # Checks if a dataframe is the type of object being passed into the data argument
    if not isinstance(data1, pd.DataFrame): 
        raise TypeError("The data argument is not of type DataFrame")
    
    if not isinstance(data2, pd.DataFrame): 
        raise TypeError("The data argument is not of type DataFrame")
        
    # Checks if the grouping column is in the dataframe
    assert grouping_col in data1.columns or data2.columns, "The grouping column does not exist in the dataframe"
    
    
    
    # Merge the 2 datasets vertically
    merged_df = pd.concat([data1, data2], axis=1)
    
    # Remove duplicated columns
    merged_df.loc[:, ~merged_df.columns.duplicated()]
    
    # Group by grouping_col and average avg_col
    grouped = merged_df.groupby(grouping_col)[avg_col].mean().sort_values(ascending = False)

    # Turn to dataframe
    grouped_df = pd.DataFrame(grouped)
    
    # reset index
    grouped_df = grouped_df.reset_index()
    
    return(grouped_df)