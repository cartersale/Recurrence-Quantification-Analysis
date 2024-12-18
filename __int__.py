# Import key functions from auto_rqa and cross_rqa
from autoRQA import autoRQA
from crossRQA import crossRQA

# Optionally import utilities for advanced users
from .utils.norm_utils import normalize_data
from .utils.plot_utils import plot_rqa_results
from .utils.rqa_utils import xRQA_dist, xRQA_stats
from .utils.output_io_utils import write_rqa_stats

# Define what the package exports
__all__ = [
    "autoRQA",
    "crossRQA",
    "normalize_data",
    "plot_rqa_results",
    "xRQA_dist",
    "xRQA_stats",
    "write_rqa_stats"
]