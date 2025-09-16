import numpy as np

def normalize_data(data, norm="none"):
    """
    Normalise data according to the specified method.

    Parameters
    ----------
    data : np.ndarray
        Input array (1D or 2D).
    norm : str or int
        Normalisation method:
            - "minmax" or 1 : rescale to [0, 1]
            - "zscore" or 2 : mean=0, std=1
            - "center" or 3 : subtract mean
            - "none"        : no normalisation

    Returns
    -------
    np.ndarray
        Normalised array.

    Raises
    ------
    ValueError
        If `norm` is not recognised.
    """
    
    # Map numeric codes to strings
    mapping = {1: "minmax", 2: "zscore", 3: "center"}
    if isinstance(norm, int):
        norm = mapping.get(norm, f"INVALID_INT_{norm}")

    if norm == "minmax":
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif norm == "zscore":
        return (data - np.mean(data)) / np.std(data)
    elif norm == "center":
        return data - np.mean(data)
    elif norm == "none":
        return data
    else:
        raise ValueError(
            f"Invalid norm option '{norm}'. "
            "Choose from: 'minmax', 'zscore', 'center', 'none' "
            "or numeric codes {1, 2, 3}."
        )
