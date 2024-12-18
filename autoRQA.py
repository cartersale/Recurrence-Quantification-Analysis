from utils import norm_utils, rqa_utils, plot_utils, output_io_utils

def autoRQA(data, params):
    """ Auto Recurrence Quantification Analysis """
    # Normalize data
    dataX = norm_utils.normalize_data(data, params['norm'])

    # Perform RQA computations
    ds = rqa_utils.xRQA_dist(dataX, dataX, params['eDim'], params['tLag'])
    td, rs, _, err_code = rqa_utils.xRQA_stats(ds['d'], params['rescaleNorm'], params['radius'], params['tmin'], params['minl'])

    ## Print stats
    if err_code == 0:
        print(f"REC: {rs['perc_recur']:.3f} | DET: {rs['perc_determ']:.3f} | MAXLINE: {rs['maxl_found']:.2f}")
    else:
        print("REC: 0.000 | DET: 0.000 | MAXLINE: 0.000")

    # Plot results
    if params['doPlots']:
        plot_utils.plot_rqa_results(
            dataX=dataX,
            td=td,
            rs=rs,
            plot_mode=params.get('plotMode', 'recurrence'),  # Default is 'recurrence'
            phase_space=params.get('phaseSpace', False),    # Default is False
            eDim=params['eDim'],
            tLag=params['tLag']
        )

    # Write stats
    if params['doStatsFile']:
        output_io_utils.write_rqa_stats("AutoRQA", params, rs, err_code)