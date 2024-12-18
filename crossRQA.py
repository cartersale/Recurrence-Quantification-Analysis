from utils import norm_utils, rqa_utils, plot_utils, output_io_utils

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
    ds = rqa_utils.xRQA_dist(dataX1, dataX2, params['eDim'], params['tLag'])
    td, rs, _, err_code = rqa_utils.xRQA_stats(ds['d'], params['rescaleNorm'], params['radius'], params['tmin'], params['minl'])

    # Print stats
    if err_code == 0:
        print(f"REC: {rs['perc_recur']:.3f} | DET: {rs['perc_determ']:.3f} | MAXLINE: {rs['maxl_found']:.2f}")
    else:
        print("REC: 0.000 | DET: 0.000 | MAXLINE: 0.000")

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