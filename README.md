# Recurrence Quantification Analysis (RQA) Python & C++ Package

This package provides fast and flexible tools for performing **Auto Recurrence Quantification Analysis (autoRQA)**, **Cross Recurrence Quantification Analysis (crossRQA)**, and **Multivariate Recurrence Quantification Analysis (multivariateRQA)** on time series data. It supports phase space reconstruction, recurrence plot generation, and computation of standard RQA metrics, including %REC, %DET, MaxLine, MeanLine, Entropy, Laminarity, Trapping Time, and more.

## Key Features

- **🚀 High-performance C++ core** with Python interface for speed
- **📊 Three analysis modes**:
  - **Traditional RQA**: Single time series with time-delay embedding
  - **Cross-RQA**: Compare two time series or systems  
  - **Multivariate RQA**: Direct analysis of multi-dimensional systems (NEW!)
- **📈 Comprehensive metrics**: All standard RQA measures plus vertical line statistics
- **🎨 Visualization tools**: Recurrence plots with optional time series overlay
- **📁 Example datasets**: Real physiological data, chaotic systems, and synthetic data
- **🛡️ Robust error handling**: Graceful handling of short time series and edge cases

## Set Up Instructions
This repository includes C++ extensions that need to be compiled before use. To compile the code, ensure you have a C++ compiler installed. Then run the following from the root directory:
```python
pip install .
```

## Parameters

The following parameters are used for **Auto RQA (`autoRQA`)**, **Cross RQA (`crossRQA`)**, and **Multivariate RQA (`multivariateRQA`)**. Note that multivariate RQA does not require embedding parameters (`eDim`, `tLag`) as it uses the actual system dimensions directly.

### Parameter List

| **Parameter**   | **Type**     | **Default** | **Description**                                                                                  |
|-----------------|--------------|-------------|--------------------------------------------------------------------------------------------------|
| **`norm`**     | `int/str`    | `1`         | Normalization method for input time series data:                                                |
|                 |              |             | - `0`: No normalization (leave data as-is).                                                    |
|                 |              |             | - `1`: Normalize to the unit interval `[0, 1]`.                                                |
|                 |              |             | - `2` or `'zscore'`: Standardize using Z-score normalization (mean = 0, std = 1).             |
|                 |              |             | - `3`: Center the data around its mean.                                                        |
| **`eDim`**     | `int`        | `3`         | **Traditional/Cross RQA only**: Embedding dimension for phase space reconstruction.            |
| **`tLag`**     | `int`        | `15`        | **Traditional/Cross RQA only**: Time lag for embedding dimensions.                             |
| **`rescaleNorm`** | `int`     | `1`         | Method for rescaling distances before thresholding:                                            |
|                 |              |             | - `1`: Rescale distances using the **mean distance**.                                          |
|                 |              |             | - `2`: Rescale distances using the **maximum distance**.                                       |
|                 |              |             | - Other: Do not rescale distances (absolute thresholding).                                     |
| **`radius`**   | `float`      | `0.1`       | Recurrence radius: Threshold value for determining recurrences. Smaller values are stricter.    |
| **`tw`**     | `int`        | `2`         | Theiler window: Minimum time separation to exclude near-diagonal recurrences (removes artifacts).|
| **`minl`**     | `int`        | `2`         | Minimum line length: The shortest line considered for calculating determinism and related metrics. |
| **`plotMode`** | `str`        | `'recurrence'` | Specifies the type of plot to generate:                                          |
|                 |              |             | - `'none'`: Do not generate plots.                              |
|                 |              |             | - `'rp'`: Basic recurrence or cross-recurrence plot only.                              |
|                 |              |             | - `'rp-timeseries'`: Plot the recurrence plot with the time series underneath or alongside. |
| **`pointSize`** | `int`     | `4`      | Size of the points in the recurrence or cross-recurrence plot.                                           |
| **`saveFig`** | `bool`     | `True`      | Whether to save the recurrence or cross-recurrence plot:                                             |
|                 |              |             | - `True`: Save plot.                                                |
|                 |              |             | - `False`: Do not save plot.                                                   |
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
import pandas as pd
from autoRQA import autoRQA

# Load example data or generate sample data
data = pd.read_csv('exampleData/PostureData.csv', header=None).iloc[:, 0].values

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
    'saveFig': False,                   # Save figure
    'showMetrics': True,                # Show metrics in the console
    'doStatsFile': False                # Write statistics to file
}

# Run Auto RQA
td, rs, mats, err_code = autoRQA(data, params)
```

### Cross Recurrence Quantification Analysis (`crossRQA`)

```python
from crossRQA import crossRQA

# Load synchronized data
data = pd.read_csv('exampleData/RockingChairData.csv', header=None)
data1 = data.iloc[:, 0].values  # First system
data2 = data.iloc[:, 1].values  # Second system

# Set parameters
params = {
    'norm': 1,                          # Normalize to unit interval
    'eDim': 3,                          # Embedding dimension
    'tLag': 15,                         # Time lag
    'rescaleNorm': 1,                   # Rescale using mean distance
    'radius': 0.1,                      # Recurrence radius
    'tw': 2,                            # Theiler window (not used in cross-RQA)
    'minl': 2,                          # Minimum line length
    'plotMode': 'rp-timeseries',        # Recurrence plot with time series
    'pointSize': 2,                     # Size of points in the plot
    'saveFig': False,                   # Save figure
    'showMetrics': True,                # Show metrics in the console
    'doStatsFile': False                # Write statistics to file
}

# Run Cross RQA
td, rs, mats, err_code = crossRQA(data1, data2, params)
```

### Multivariate Recurrence Quantification Analysis (`multivariateRQA`) - NEW!

```python
from multivariateRQA import multivariateRQA, multivariateCrossRQA

# Load multivariate data (e.g., 3D Lorenz chaotic system)
data = pd.read_csv('exampleData/lorenz_chaotic_xyz.csv')
multivar_data = data[['x', 'y', 'z']].values  # Use actual system dimensions

# Set parameters (no embedding needed!)
params = {
    'norm': 'zscore',                   # Z-score normalization
    'rescaleNorm': 1,                   # Rescale using mean distance
    'radius': 0.15,                     # Recurrence radius
    'tw': 1,                            # Theiler window for auto-RQA
    'minl': 2,                          # Minimum line length
    'plotMode': 'rp',                   # Recurrence plot
    'pointSize': 1,                     # Size of points in the plot
    'saveFig': False,                   # Save figure
    'showMetrics': True,                # Show metrics in the console
    'doStatsFile': False                # Write statistics to file
}

# Run Multivariate Auto-RQA
td, rs, mats, err_code = multivariateRQA(multivar_data, params)

# Or run Multivariate Cross-RQA between two systems
data1 = pd.read_csv('exampleData/lorenz_chaotic_xyz.csv')[['x', 'y', 'z']].values
data2 = pd.read_csv('exampleData/lorenz_chaotic_xyz_2.csv')[['x', 'y', 'z']].values
td, rs, mats, err_code = multivariateCrossRQA(data1, data2, params)
```

### Key Advantages of Multivariate RQA:
- **No embedding parameters** (`eDim`, `tLag`) - uses actual system dimensions
- **Captures true coupling** between variables
- **Ideal for**: Coupled oscillators, multi-channel physiological signals, climate data
- **Direct analysis** of known multi-dimensional systems

---

## Example Datasets

The `exampleData/` folder contains diverse time series for testing and learning RQA methods:

### Physiological & Experimental Data:
- **PostureData.csv**: Human postural sway (balance) data
- **RockingChairData.csv**: Interpersonal coordination between rocking chairs
- **Elvis.csv**: Categorical/discrete state transitions

### Synthetic & Control Data:
- **WhiteNoiseData.csv**: Random signals for baseline comparison

### Chaotic Systems (ideal for Multivariate RQA):
- **lorenz_chaotic_xyz.csv**: 3D Lorenz attractor (classic chaotic system)
- **lorenz_chaotic_xy.csv**: 2D subset for testing
- **lorenz_chaotic_xyz_2.csv**: Second trajectory for cross-RQA
- **lorenz_hyperchaotic_xyz.csv**: More complex dynamics

See `exampleData/README.md` for detailed descriptions and usage examples for each dataset.

## Quick Start with Jupyter Notebook

Run the `analysisExamples.ipynb` notebook to see all three RQA methods in action with real data examples, including:
- Traditional RQA on postural sway data
- Cross-RQA on interpersonal coordination
- Categorical RQA on discrete states
- **Multivariate RQA on 3D chaotic systems** (demonstrates the new functionality)

---

## Related Resources

For guidance on using RQA to explore human behaviour in social and behavioural research: 

- Chapter: Richardson, M. J., Dale, R., & Marsh, K. L. (2014). Complex dynamical systems in social and personality psychology. *Handbook of research methods in social and personality psychology*, pg. 253.

- Chapter: Macpherson, C., Richardson, M., & Kallen, R. W. (2024). Advanced quantitative approaches: Linear and non-linear time-series analyses. In *Cambridge handbook of research methods and statistics for the social and behavioral sciences* (Vol. 3). Cambridge University Press (CUP).

- Python tutorial: [Linear and Nonlinear Time Series Analysis](https://github.com/xkiwilabs/Linear-NonLinear-TSAnalysis)

## References

This code is based on a [matlab toolbox](https://github.com/xkiwilabs/MATLAB-Toolboxes/tree/master/RQAToolbox) developed by Bruce Kay and Mike Richardson, with contributions from countless collaborators. 
