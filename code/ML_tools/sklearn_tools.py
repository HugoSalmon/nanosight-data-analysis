

import numpy as np



def kolmogorov_smirnov_test_statistic(u_values, v_values, u_weights=None, v_weights=None):
    r"""
    Compute, between two one-dimensional distributions :math:`u` and
    :math:`v`, whose respective CDFs are :math:`U` and :math:`V`, the
    Kolmogorov Smirnov test statistic
    Parameters
    ----------
    u_values, v_values : array_like
        Values observed in the (empirical) distribution.
    u_weights, v_weights : array_like, optional
        Weight for each value. If unspecified, each value is assigned the same
        weight.
        `u_weights` (resp. `v_weights`) must have the same length as
        `u_values` (resp. `v_values`). If the weight sum differs from 1, it
        must still be positive and finite so that the weights can be normalized
        to sum to 1.
    Returns
    -------
    distance : float
        The computed distance between the distributions.
    Notes
    -----
    The input distributions can be empirical, therefore coming from samples
    whose values are effectively inputs of the function, or they can be seen as
    generalized functions, in which case they are weighted sums of Dirac delta
    functions located at the specified values.
    """
    
    
    
    u_values, u_weights = _validate_distribution(u_values, u_weights)
    v_values, v_weights = _validate_distribution(v_values, v_weights)

    u_sorter = np.argsort(u_values)
    v_sorter = np.argsort(v_values)

    all_values = np.concatenate((u_values, v_values))
    all_values.sort(kind='mergesort')

    # Compute the differences between pairs of successive values of u and v.
    deltas = np.diff(all_values)

    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
    v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')

    # Calculate the CDFs of u and v using their weights, if specified.
    if u_weights is None:
        u_cdf = u_cdf_indices / u_values.size
    else:
        u_sorted_cumweights = np.concatenate(([0],
                                              np.cumsum(u_weights[u_sorter])))
        u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

    if v_weights is None:
        v_cdf = v_cdf_indices / v_values.size
    else:
        v_sorted_cumweights = np.concatenate(([0],
                                              np.cumsum(v_weights[v_sorter])))
        v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]


    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1)
    # ax.plot(u_cdf)
    # ax.plot(v_cdf)
    
    # fig, ax = plt.subplots(1)
    # ax.plot(u_values, u_weights)
    # ax.plot(v_values, v_weights)
    

    
    return np.max(np.abs(u_cdf-v_cdf))



def _validate_distribution(values, weights):
    """
    Validate the values and weights from a distribution input of `cdf_distance`
    and return them as ndarray objects.
    Parameters
    ----------
    values : array_like
        Values observed in the (empirical) distribution.
    weights : array_like
        Weight for each value.
    Returns
    -------
    values : ndarray
        Values as ndarray.
    weights : ndarray
        Weights as ndarray.
    """
    # Validate the value array.
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("Distribution can't be empty.")

    # Validate the weight array, if specified.
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(values):
            raise ValueError('Value and weight array-likes for the same '
                             'empirical distribution must be of the same size.')
        if np.any(weights < 0):
            raise ValueError('All weights must be non-negative.')
        if not 0 < np.sum(weights) < np.inf:
            raise ValueError('Weight array-like sum must be positive and '
                             'finite. Set as None for an equal distribution of '
                             'weight.')

        return values, weights

    return values, None
