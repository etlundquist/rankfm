"""
rankfm general utility functions
"""

def get_data(obj):
    """get the numeric data from either a pd.dataframe or np.ndarray

    :param obj: pd.dataframe or np.ndarray
    :return: the object's underlying np.ndarray data
    """

    if obj.__class__.__name__ in ('DataFrame', 'Series'):
        data = obj.values
    elif obj.__class__.__name__ == 'ndarray':
        data = obj
    else:
        raise TypeError("input data must be in either pd.dataframe/pd.series or np.ndarray format")
    return data
