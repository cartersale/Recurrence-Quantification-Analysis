from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def plot_rqa_results(dataX, dataY=None, td=None, rs=None, plot_mode='recurrence', phase_space=False, eDim=3, tLag=1):
    """
    Plot RQA results based on the chosen mode.

    Parameters:
        dataX (np.ndarray): Time series data for X-axis.
        dataY (np.ndarray): Time series data for Y-axis (for cross-RQA).
        td (np.ndarray): Thresholded distance matrix (recurrence or cross-recurrence plot).
        rs (dict): RQA statistics dictionary (optional for display).
        plot_mode (str): 
            'recurrence': Basic recurrence plot.
            'recurrence_with_timeseries': Recurrence plot with time series underneath or to the side.
            'phase_space': Recurrence plot with phase space reconstruction.
        phase_space (bool): True to include a 2D/3D phase space plot.
        eDim (int): Embedding dimension for phase space.
        tLag (int): Time lag for phase space.
    """

    def reconstruct_phase_space(data, dim, lag):
        """ Reconstruct phase space using embedding dimension and time lag """
        n_points = len(data) - (dim - 1) * lag
        phase_space = np.array([data[i:i + n_points] for i in range(0, dim * lag, lag)]).T
        return phase_space

    plt.figure(figsize=(10, 8))

    # Plot 1: Recurrence or Cross-Recurrence Plot
    plt.subplot(2, 2, 1)
    plt.imshow((1 - td).T, cmap='gray', origin='lower')
    title = "Cross-Recurrence Plot" if dataY is not None else "Recurrence Plot"
    plt.title(title)
    plt.xlabel("X(i)")
    plt.ylabel("Y(j)" if dataY is not None else "X(j)")

    # Optionally display RQA statistics
    if rs:
        plt.figtext(0.5, 0.02, f"%REC: {rs['perc_recur']:.2f} | %DET: {rs['perc_determ']:.2f} | "
                               f"MAXLINE: {rs['maxl_found']:.0f} | MEANLINE: {rs['llmnsd'][0]:.0f} | "
                               f"ENTROPY: {rs['entropy'][0]:.2f}",
                    ha="center", fontsize=10)

    # Plot 2: Time Series - Underneath or to the side
    if plot_mode == 'recurrence_with_timeseries':
        plt.subplot(2, 2, 3)
        plt.plot(dataX, 'b-')
        plt.title("Time Series (X)")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        if dataY is not None:
            plt.subplot(2, 2, 4)
            plt.plot(dataY, 'g-')
            plt.title("Time Series (Y)")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")

    # Plot 3: Phase Space Reconstruction
    if phase_space:
        phase_data = reconstruct_phase_space(dataX, eDim, tLag)
        if eDim == 2:
            plt.subplot(2, 2, 4)
            plt.plot(phase_data[:, 0], phase_data[:, 1], 'b-')
            plt.title("2D Phase Space Reconstruction")
            plt.xlabel("X(t)")
            plt.ylabel(f"X(t + {tLag})")
        elif eDim >= 3:
            ax = plt.subplot(2, 2, 4, projection='3d')
            ax.plot(phase_data[:, 0], phase_data[:, 1], phase_data[:, 2], 'b-')
            ax.set_title("3D Phase Space Reconstruction")
            ax.set_xlabel("X(t)")
            ax.set_ylabel(f"X(t + {tLag})")
            ax.set_zlabel(f"X(t + {2 * tLag})")

    plt.tight_layout()
    plt.show()