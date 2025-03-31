# Recurrence Quantification Analysis (RQA) Package

## Parameters Documentation

The following parameters are used for both **Auto Recurrence Quantification Analysis (`autoRQA`)** and **Cross Recurrence Quantification Analysis (`crossRQA`)**.

### Parameter List

| **Parameter**   | **Type**     | **Default** | **Description**                                                                                  |
|-----------------|--------------|-------------|--------------------------------------------------------------------------------------------------|
| **`norm`**     | `int`        | `1`         | Normalization method for input time series data:                                                |
|                 |              |             | - `0`: No normalization (leave data as-is).                                                    |
|                 |              |             | - `1`: Normalize to the unit interval `[0, 1]`.                                                |
|                 |              |             | - `2`: Standardize using Z-score normalization (mean = 0, std = 1).                             |
|                 |              |             | - `3`: Center the data around its mean.                                                        |
| **`eDim`**     | `int`        | `3`         | Embedding dimension: Specifies the number of dimensions to reconstruct the phase space.         |
| **`tLag`**     | `int`        | `15`        | Time lag: Specifies the delay between successive embedding dimensions.                         |
| **`rescaleNorm`** | `int`     | `1`         | Method for rescaling distances before thresholding:                                            |
|                 |              |             | - `1`: Rescale distances using the **mean distance**.                                          |
|                 |              |             | - `2`: Rescale distances using the **maximum distance**.                                       |
|                 |              |             | - Other: Do not rescale distances (absolute thresholding).                                     |
| **`radius`**   | `float`      | `0.1`       | Recurrence radius: Threshold value for determining recurrences. Smaller values are stricter.    |
| **`tw`**     | `int`        | `2`         | Theiler window: Minimum time separation to exclude near-diagonal recurrences (removes artifacts).|
| **`minl`**     | `int`        | `2`         | Minimum line length: The shortest line considered for calculating determinism and related metrics. |
| **`plotMode`** | `str`        | `'recurrence'` | Specifies the type of plot to generate:                                          |
|                 |              |             | - `'none'`: Do not generate plots.                              |
|                 |              |             | - `'recurrence'`: Basic recurrence or cross-recurrence plot only.                              |
|                 |              |             | - `'recurrence_with_timeseries'`: Plot the recurrence plot with the time series underneath or alongside. |
| **`pointSize`** | `int`     | `4`      | Size of the points in the recurrence or cross-recurrence plot.                                           |
| **`showMetrics`** | `bool`     | `True`      | Whether to show RQA statistics in the console:                                             |
|                 |              |             | - `True`: Show metrics in the console.                                                |
|                 |              |             | - `False`: Do not show metrics in the console.                                                   |
| **`doStatsFile`** | `bool`    | `False`     | Whether to write RQA statistics to a file (`RQA_Stats.csv`):                                   |
|                 |              |             | - `True`: Write the statistics to a file.                                                   |
|                 |              |             | - `False`: Do not write statistics to a file.                                                 |

---

## Example Usage

### Auto Recurrence Quantification Analysis (`autoRQA`)

```python
import numpy as np
from rqa_analysis import autoRQA

# Generate sample time series data
data = np.sin(np.linspace(0, 10 * np.pi, 300))

# Set parameters
params = {
    'norm': 1,                          # Normalize to unit interval
    'eDim': 3,                          # Embedding dimension
    'tLag': 15,                         # Time lag
    'rescaleNorm': 1,                   # Rescale using mean distance
    'radius': 0.1,                      # Recurrence radius
    'tw': 2,                            # Theiler window
    'minl': 2,                          # Minimum line length
    'plotMode': 'rp-timeseries',        # Recurrence plot with time series
    'pointSize': 2,                     # Size of points in the plot
    'showMetrics': True,                # Show metrics in the console
    'doStatsFile': True                 # Write statistics to file
}

# Run Auto RQA
autoRQA(data, params)
```

### Cross Recurrence Quantification Analysis (`crossRQA`)

```python
from rqa_analysis import crossRQA

# Generate sample time series data
data1 = np.sin(np.linspace(0, 10 * np.pi, 300))
data2 = np.cos(np.linspace(0, 10 * np.pi, 300))

# Set parameters
params = {
    'norm': 1,                          # Normalize to unit interval
    'eDim': 3,                          # Embedding dimension
    'tLag': 15,                         # Time lag
    'rescaleNorm': 1,                   # Rescale using mean distance
    'radius': 0.1,                      # Recurrence radius
    'tw': 2,                            # Theiler window
    'minl': 2,                          # Minimum line length
    'plotMode': 'rp-timeseries',        # Recurrence plot with time series
    'pointSize': 2,                     # Size of points in the plot
    'showMetrics': True,                # Show metrics in the console
    'doStatsFile': True                 # Write statistics to file
}

# Run Cross RQA
crossRQA(data1, data2, params)
```

---

## Parameter Highlights

1. **`tw` and `minl`**:
   - Adjust these parameters to filter out diagonal artifacts and short diagonal lines, which may arise from noise or trivial recurrences.

2. **`radius`**:
   - This controls how "close" points in phase space need to be to count as recurrent. Smaller radii result in sparser recurrence plots.

3. **`plotMode`**:
   - Flexible visualization: Users can choose basic recurrence plots, add time series, or include phase space reconstructions.

4. **`doStatsFile`**:
   - Enables automatic output of RQA statistics into a CSV file for further analysis.
