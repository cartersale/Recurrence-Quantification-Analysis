import numpy as np
import scipy.stats as stats

def xRQA_dist(a, b, dim, lag):
    """
    Compute distances between all points of two vectors,
    which can be embedded using time-lags.

    Parameters:
        a (np.ndarray): Data vector 1xN or Nx1.
        b (np.ndarray): Data vector 1xN or Nx1.
        dim (int): Embedding dimension (scalar).
        lag (int): Time lag in samples (scalar).

    Returns:
        ds (dict): A dictionary containing:
            - 'dim': Embedding dimension
            - 'lag': Time lag
            - 'd': Distance matrix
    """
    # Convert input to numpy arrays and ensure they are column vectors
    a = np.asarray(a).reshape(-1, 1)
    b = np.asarray(b).reshape(-1, 1)

    # Check embedding size
    n = a.shape[0]
    n2 = n - lag * (dim - 1)
    if n2 <= 0:
        raise ValueError("Not enough data for these embedding parameters.")

    # Initialize distance matrix
    dist = np.zeros((n2, n2), dtype=np.float32)

    # Calculate Distance Matrix
    if dim > 1:
        emb_a = np.zeros((n2, dim), dtype=np.float32)
        emb_b = np.zeros((n2, dim), dtype=np.float32)
        for k in range(dim):
            emb_a[:, k] = a[lag * k: lag * k + n2, 0]
            emb_b[:, k] = b[lag * k: lag * k + n2, 0]

        for i in range(n2):
            v = emb_a[i, :] - emb_b
            dist[i, :] = np.sqrt(np.sum(v ** 2, axis=1))
    else:
        for i in range(n2):
            dist[i, :] = np.abs(a[i, 0] - b[:n2, 0])

    # Return as dictionary
    ds = {'dim': dim, 'lag': lag, 'd': dist}
    return ds



def xRQA_radius(dist, rescale, rad, tmin):
    """
    Finds all points in a distance matrix less than or equal to a given radius.
    Used for recurrence plots.

    Parameters:
        dist (np.ndarray): Square (NxN) distance matrix, with distances only on the upper diagonal.
        rescale (int): 
            1 = Use mean distance to rescale
            2 = Use max distance to rescale
            Other = Don't rescale (absolute option)
        rad (float): Distance threshold (scalar).
        tmin (int): Minimum recurrence time to exclude diagonals within this window.

    Returns:
        thrd (np.ndarray): Thresholded distance matrix (NxN).
                          1 where distance <= rad, 0 otherwise.
                          Distances within the Theiler window (tmin) are set to 0.
                          Returns -1 on input errors.
    """

    # Check input arguments
    if dist.shape[0] != dist.shape[1]:
        print("Distance matrix must be square")
        return -1
    if dist.size == 1:
        print("Distance matrix has only one element!")
        return -1
    if not isinstance(rad, (int, float)) or rad <= 0:
        print("Please use a scalar threshold > 0")
        return -1
    if not isinstance(rescale, int):
        print("Please use a scalar rescale parameter")
        return -1
    if not isinstance(tmin, int) or tmin < 0:
        print("Please use an integer tmin >= 0")
        return -1

    # Rescale distance matrix
    if rescale == 1:
        mean_dist = np.mean(dist)
        dist = dist / mean_dist
    elif rescale == 2:
        max_dist = np.max(dist)
        dist = dist / max_dist

    # Initialize thresholded matrix
    ldist = dist.shape[0]
    thrd = np.zeros((ldist, ldist), dtype=np.int8)

    # Thresholding
    for i in range(ldist):
        for j in range(ldist):
            if dist[i, j] <= rad:
                thrd[i, j] = 1

    # Apply Theiler window (set values within tmin diagonals to 0)
    if tmin != 0:
        for i in range(tmin):
            for j in range(ldist - i):
                thrd[j, j + i] = 0  # Upper diagonal
                thrd[j + i, j] = 0  # Lower diagonal

    return thrd



def xRQA_line(thrd, tmin):
    """
    Find all diagonal lines (parallel to the main diagonal) and their lengths 
    in a thresholded distance matrix. Computes trend measures as well.

    Parameters:
        thrd (np.ndarray): Square thresholded distance matrix with 1's and 0's.
        tmin (int): Minimum recurrence time (Theiler window).

    Returns:
        ll (np.ndarray): Vector containing the lengths of all diagonal lines found.
        maxl_poss (int): Maximum possible line length.
        npts (int): Total number of possible points in the recurrence plot (excluding the Theiler window).
        trend1 (float): Trend measure for the lower triangle.
        trend2 (float): Trend measure for the upper triangle.
    """
    # Input validation
    if thrd.shape[0] != thrd.shape[1]:
        print("Thresholded distance matrix must be square")
        return 0, 0, 0, 0, 0
    if np.max(thrd) > 1:
        print("Please use a thresholded distance matrix with 1's and 0's")
        return 0, 0, 0, 0, 0
    if not isinstance(tmin, int) or tmin < 0:
        print("Please use an integer tmin >= 0")
        return 0, 0, 0, 0, 0

    n = thrd.shape[0]
    possnumll = (n**2) // 2
    ll = np.zeros(possnumll, dtype=np.int16)  # Storage for line lengths
    recur = np.zeros((2 * n - 1, 2), dtype=np.float32)  # To store diagonal recurrences
    nlines = 0

    # Process diagonals
    for i in range(2 * n - 1):  # For all diagonals
        d = np.diag(thrd, i - n + 1)  # Extract diagonal
        ld = len(d)
        recur[i, 0] = ld  # Total length of the diagonal
        j = 0
        while j < ld:
            if d[j] == 1:
                nlines += 1
                ll[nlines - 1] = 1
                recur[i, 1] += 1
                k = j + 1
                while k < ld and d[k] == 1:
                    ll[nlines - 1] += 1
                    recur[i, 1] += 1
                    k += 1
                j = k
            else:
                j += 1

    # Trim ll to actual number of lines found
    ll = ll[:nlines]

    # Compute trends
    mid = np.argmax(recur[:, 0])
    recur[:, 1] = recur[:, 1] / recur[:, 0]  # Normalize by diagonal length

    # Lower diagonal trend
    first = tmin
    last = n - 1
    indx = np.arange(first, last + 1)
    y_lower = 100 * recur[mid - tmin::-1, 1][:len(indx)]
    trend1_slope, _, _, _, _ = stats.linregress(indx, y_lower)
    trend1 = 1000 * trend1_slope

    # Upper diagonal trend
    first = mid + tmin
    last = 2 * n - 2
    indx = np.arange(tmin, last - mid + 1)
    y_upper = 100 * recur[first:last + 1, 1][:len(indx)]
    trend2_slope, _, _, _, _ = stats.linregress(indx, y_upper)
    trend2 = 1000 * trend2_slope

    # Maximum possible line length
    maxl_poss = n - tmin

    # Total number of points excluding Theiler window
    if tmin == 0:
        npts = n**2
    else:
        npts = n**2 - n - 2 * n * (tmin - 1) + tmin * (tmin - 1)

    return ll, maxl_poss, npts, trend1, trend2



def xRQA_histlines(llengths, minl):
    """
    Compute the histogram of line lengths and their statistics.

    Parameters:
        llengths (list or np.ndarray): Vector of line lengths.
        minl (int): Minimum line length to consider.

    Returns:
        linehist (np.ndarray): Nx2 matrix where:
            - Column 1: Line lengths
            - Column 2: Frequency of each line length
        linestats (list): [mean, standard deviation, count] for line lengths >= minl.
    """

    # Input validation
    llengths = np.array(llengths).flatten()
    if llengths.ndim != 1:
        print("Input data must be a vector, not a matrix")
        return -1, [-1, -1, -1]
    if not isinstance(minl, int) or minl <= 0:
        print("Please use an integer min line length >= 1")
        return -1, [-1, -1, -1]

    # Filter line lengths greater than or equal to minl
    valid_lengths = llengths[llengths >= minl]

    if valid_lengths.size == 0:
        # No valid line lengths
        return np.array([[0, 0]]), [0, 0, 0]

    # Calculate statistics
    ll_mean = np.mean(valid_lengths)
    ll_std = np.std(valid_lengths)
    ll_n = len(valid_lengths)
    linestats = [ll_mean, ll_std, ll_n]

    # Create histogram of line lengths
    unique_lengths, counts = np.unique(valid_lengths, return_counts=True)
    linehist = np.vstack((unique_lengths, counts)).T.astype(np.float32)

    return linehist, linestats

import numpy as np

def xRQA_entropy(distr, nstates):
    """
    Compute Shannon entropy of a distribution.

    Parameters:
        distr (list or np.ndarray): Distribution vector (1D array).
        nstates (int): Number of possible states.

    Returns:
        entropy (list): [Shannon entropy, remaining information].
                        Returns -1 on input errors.
    """
    # Input validation
    distr = np.asarray(distr).flatten()
    if distr.ndim != 1:
        print("Input data must be a vector, not a matrix")
        return -1
    if not isinstance(nstates, int) or nstates <= 0:
        print("Please use an integer greater than 0 for the number of states")
        return -1

    # Normalize the distribution
    distr_sum = np.sum(distr)
    if distr_sum == 0:
        print("Sum of the distribution is zero; invalid input.")
        return -1
    distr2 = distr / distr_sum

    # Compute Shannon entropy
    shannon_entropy = -np.sum(distr2[distr2 > 0] * np.log2(distr2[distr2 > 0]))

    # Compute maximum entropy and remaining information
    max_entropy = np.log2(nstates)
    remaining_info = max_entropy - shannon_entropy

    # Return entropy values
    return [shannon_entropy, remaining_info]



def xRQA_stats(d, rescale, rad, tmin, minl):
    """
    Perform Recurrence Quantification Analysis (RQA) on a distance matrix.

    Parameters:
        d (np.ndarray): Distance matrix.
        rescale (int): Rescale option (1 = mean, 2 = max, other = absolute).
        rad (float): Threshold radius.
        tmin (int): Minimum recurrence time (Theiler window).
        minl (int): Minimum line length for determinism, histogram, and entropy.

    Returns:
        td (np.ndarray): Thresholded distance matrix.
        rs (dict): Structure containing RQA statistics.
        mats (dict): Structure containing intermediate matrices and results.
        err_code (int): Error code (0 if successful, >0 if error occurred).
    """
    err_code = 0

    # Threshold the distance matrix
    td = xRQA_radius(d, rescale, rad, tmin)
    if isinstance(td, int) and td == -1:
        print("Error in thresholding.")
        return None, None, None, 1

    # Find all lines and compute trends
    ll, maxl_poss, npts, trend1, trend2 = xRQA_line(td, tmin)
    if isinstance(ll, int) and ll == 0:
        print("Error in line counting.")
        return None, None, None, 2

    # Compute histogram of line lengths
    lh, llmnsd = xRQA_histlines(ll, minl)
    if isinstance(lh, int) and lh == -1:
        print("Error in creating line histograms.")
        return None, None, None, 3

    # Compute entropy of line length histogram
    if lh.size > 1:  # Check if histogram has data
        entropy = xRQA_entropy(lh[:, 1], maxl_poss - minl + 1)
        if isinstance(entropy, int) and entropy == -1:
            print("Error in computing entropy.")
            return None, None, None, 4
    else:
        entropy = [0, 0]

    # Percent recurrence
    recur = np.sum(ll)
    perc_rec = 100 * recur / npts

    # Determinism, Max line, and Complexity
    if lh.size > 1:
        perc_determ = 100 * np.sum(lh[:, 0] * lh[:, 1]) / recur  # % Determinism
        maxl_found = np.max(lh[:, 0])  # Maximum line length found
    else:
        perc_determ = 0
        maxl_found = 0

    # Output results into structures
    rs = {
        'rescale': rescale,
        'rad': rad,
        'tmin': tmin,
        'minl': minl,
        'perc_recur': perc_rec,
        'perc_determ': perc_determ,
        'npts': npts,
        'entropy': entropy,
        'maxl_poss': maxl_poss,
        'maxl_found': maxl_found,
        'trend1': trend1,
        'trend2': trend2,
        'llmnsd': llmnsd
    }

    mats = {
        'rescale': rescale,
        'rad': rad,
        'tmin': tmin,
        'minl': minl,
        'td': td,
        'll': ll,
        'lh': lh
    }

    return td, rs, mats, err_code
