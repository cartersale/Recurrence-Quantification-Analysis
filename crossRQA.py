from utils import norm_utils, plot_utils, output_io_utils
from utils import rqa_utils_cpp

def crossRQA(data1, data2, params):
    """
    Perform Cross Recurrence Quantification Analysis (RQA).
    Parameters:
        data1 (np.ndarray): Time series data 1.
        data2 (np.ndarray): Time series data 2.
        params (dict): Dictionary of RQA parameters:
                       - norm, eDim, tLag, rescaleNorm, radius, tmin, minl,
                         doPlots, plotMode, phaseSpace, doStatsFile
    """

    # Normalize data
    dataX1 = norm_utils.normalize_data(data1, params['norm'])
    dataX2 = norm_utils.normalize_data(data2, params['norm'])

    # Perform RQA computations
    ds = rqa_utils_cpp.rqa_dist(dataX1, dataX2, dim=params['eDim'], lag=params['tLag'])

    # Similarly, you can call xRQA_stats:
    td, rs, mats, err_code = rqa_utils_cpp.rqa_stats(ds["d"], rescale=params['rescaleNorm'], rad=params['radius'], diag_ignore=0, minl=params['minl'], rqa_mode="cross")

    # Print stats
    if err_code == 0:
        print(f"REC: {float(rs['perc_recur']):.3f} | DET: {float(rs['perc_determ']):.3f} | MAXLINE: {float(rs['maxl_found']):.2f}")
        print(f"Lines (mean): {float(rs['llmnsd'][0]):.2f} | Lines (sd): {float(rs['llmnsd'][2]):.2f} | Lines (count): {float(rs['llmnsd'][2]):.2f}")
        print(f"ENTR: {float(rs['entropy'][0]):.3f} | LAM: {float(rs['laminarity']):.3f} | TT: {float(rs['trapping_time']):.3f}")
        print(f"Vmax: {float(rs['vmax']):.2f} | Divergence: {float(rs['divergence']):.3f} | TREND: {float(rs['trend1']):.3f} and {float(rs['trend2']):.3f}")
    else:
        print("Error in RQA computation. Check parameters and data.")

    # Plot results
    if params['doPlots']:
        plot_utils.plot_rqa_results(
            dataX=dataX1,
            dataY=dataX2,
            td=td,
            rs=rs,
            plot_mode=params.get('plotMode', 'recurrence'),  # Default is 'recurrence'
            phase_space=params.get('phaseSpace', False),    # Default is False
            eDim=params['eDim'],
            tLag=params['tLag']
        )

    # Write stats
    if params['doStatsFile']:
        output_io_utils.write_rqa_stats("CrossRQA", params, rs, err_code)