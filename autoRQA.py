from utils import norm_utils, plot_utils, output_io_utils
from utils import rqa_utils_cpp

def autoRQA(data, params):
    """ Auto Recurrence Quantification Analysis """
    # Normalize data
    dataX = norm_utils.normalize_data(data, params['norm'])

    # Perform RQA computations
    ds = rqa_utils_cpp.rqa_dist(dataX, dataX, dim=params['eDim'], lag=params['tLag'])

    # Similarly, you can call xRQA_stats:
    td, rs, mats, err_code = rqa_utils_cpp.rqa_stats(ds["d"], rescale=params['rescaleNorm'], rad=params['radius'], diag_ignore=params['tw'], minl=params['minl'], rqa_mode="auto")

    ## Print stats
    if err_code == 0:
        if params['showMetrics']:
            print(f"%REC: {float(rs['perc_recur']):.3f} | %DET: {float(rs['perc_determ']):.3f} | MaxLine: {float(rs['maxl_found']):.2f}")
            print(f"Mean Line Length: {float(rs['mean_line_length']):.2f} | SD Line Length: {float(rs['std_line_length']):.2f} | Line Count): {float(rs['count_line']):.2f}")
            print(f"ENTR: {float(rs['entropy']):.3f} | LAM: {float(rs['laminarity']):.3f} | TT: {float(rs['trapping_time']):.3f}")
            print(f"Vmax: {float(rs['vmax']):.2f} | Divergence: {float(rs['divergence']):.3f}") 
            print(f"Trend_Lower: {float(rs['trend_lower_diag']):.3f} | Trend_Upper {float(rs['trend_upper_diag']):.3f}")
    else:
        print("Error in RQA computation. Check parameters and data.")

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