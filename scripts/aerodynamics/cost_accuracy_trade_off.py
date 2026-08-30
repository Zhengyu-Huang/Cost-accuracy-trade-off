"""Plot distributional OpenFOAM and M-PCNO cost--accuracy summaries.

Run from ``scripts/aerodynamics``::

    python cost_accuracy_trade_off.py

Input ``data/cost_accuracy_mpcno_solver_data.npz`` must have been produced by
``mpcno_solver.py``. The six OpenFOAM cases are embedded below as literal cost,
runtime, and error arrays. The active ``__main__`` calls
``cost_accuracy_plot()`` and writes ``figs/aerodynamics_cost_accuracy.png``.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FixedLocator, NullLocator

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 28,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2.4
})
formatter = ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
formatter.set_powerlimits((-2, 2))
formatter.set_useOffset(True)
lbl = "#000000"
tk = "#808080"
    



def compute_cost_traditional_solver(ne, nt, nt_sub=2):
    """Evaluate the manuscript's per-cell FVM flop model."""
    return nt*(1212*ne + 168*nt_sub*ne)

def cost_accuracy_plot():
    """Combine saved neural results with the six-case OpenFOAM summary."""
    # Neural arrays are [resolution, case, metric]. Cost metrics are FLOPs,
    # CPU seconds, and GPU seconds; accuracy metrics are relative L2 and L1.
    cost_accuracy_mpcno_solver_data = np.load('data/cost_accuracy_mpcno_solver_data.npz', allow_pickle=True)
    cost_mpcno_solver     = cost_accuracy_mpcno_solver_data['cost']        # floating point, cpu, gpu
    accuracy_mpcno_solver = cost_accuracy_mpcno_solver_data['accuracy']    # rel l2, rel l1

    # Rows correspond to the six listed geometries; columns follow
    # [large reference, large, medium, small]. For each geometry, the first
    # row stores estimated FLOPs and the second stores measured CPU seconds.
    # "drivaerFastback", "E_S_WWC_WM_005", "E_S_WW_WM_001",
    # "N_S_WWC_WM_001", "F_S_WWC_WM_001.stl", "N_S_WW_WM_001"
    cost_traditional_solver = np.array([[[compute_cost_traditional_solver(22551532, 7000), compute_cost_traditional_solver(22551532, 2000), compute_cost_traditional_solver(2982335, 1000), compute_cost_traditional_solver(440405, 1000)],
                                         [6424, 1868, 975, 557]],
                                        [[compute_cost_traditional_solver(23380779, 7000), compute_cost_traditional_solver(23380779, 2000), compute_cost_traditional_solver(3109094, 1000), compute_cost_traditional_solver(463955, 1000)],
                                         [6728, 1934, 885, 685]],
                                        [[compute_cost_traditional_solver(22139572, 7000), compute_cost_traditional_solver(22139572, 2000), compute_cost_traditional_solver(2930129 , 1000), compute_cost_traditional_solver(436961, 1000)],
                                         [6445, 1846, 914, 449]],
                                        [[compute_cost_traditional_solver(22073620, 7000), compute_cost_traditional_solver(22073620, 2000), compute_cost_traditional_solver(2919147, 1000), compute_cost_traditional_solver(435478, 1000)],
                                         [6339, 1842, 909, 512]],
                                        [[compute_cost_traditional_solver(22122942, 7000), compute_cost_traditional_solver(22122942, 2000), compute_cost_traditional_solver(2918778, 1000), compute_cost_traditional_solver(435347, 1000)],
                                         [6310, 1813, 943 ,517]],                                        
                                        [[compute_cost_traditional_solver(22139572, 7000), compute_cost_traditional_solver(22139572, 2000), compute_cost_traditional_solver(2918103, 1000), compute_cost_traditional_solver(434952, 1000)],
                                         [6417, 1858, 905 ,562]]
                                        ])
    # OpenFOAM errors use each geometry's large, 7000-iteration result as zero.
    accuracy_traditional_solver = np.array( [[0, 0.11461073386621388, 0.16311370793394878, 0.23627941049202983],
                                             [0, 0.1197486562006796,  0.1639763541317384,  0.25645566161729383],
                                             [0, 0.11019880045475844, 0.1546867948223228,  0.21574105608758942],
                                             [0, 0.13475928881804267, 0.186263260252123,   0.2962403338662556],
                                             [0, 0.130290288727653,   0.22925902482126198, 0.3041899192569055],
                                             [0, 0.11415673601694837, 0.1944082320327314,  0.244779562992878]])

    # Aggregate over six OpenFOAM geometries. Neural summaries below aggregate
    # over 512 different geometries, so the plotted comparison is not paired.
    mean_cost_traditional_solver = np.mean(cost_traditional_solver, axis=0)  
    mean_accuracy_traditional_solver = np.mean(accuracy_traditional_solver, axis=0)    
    std_cost_traditional_solver  = np.std(cost_traditional_solver, axis=0, ddof=1)       
    std_accuracy_traditional_solver  = np.std(accuracy_traditional_solver, axis=0, ddof=1)

    print("mean_cost_traditional_solver: ",    mean_cost_traditional_solver)
    print("mean_accuracy_traditional_solver:", mean_accuracy_traditional_solver)

    
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    for ax in axs:
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.6)
    

    
    mean_cost_mpcno_solver = np.mean(cost_mpcno_solver, axis=1)  
    mean_accuracy_mpcno_solver = np.mean(accuracy_mpcno_solver, axis=1)    
    std_cost_mpcno_solver  = np.std(cost_mpcno_solver, axis=1, ddof=1)       
    std_accuracy_mpcno_solver  = np.std(accuracy_mpcno_solver, axis=1, ddof=1)

    print("mean_cost_mpcno_solver: ",    mean_cost_mpcno_solver)
    print("mean_accuracy_mpcno_solver:", mean_accuracy_mpcno_solver)

    # axs[0].loglog(mean_accuracy_traditional_solver, mean_cost_traditional_solver[...,0], 'o-')
    axs[0].errorbar(mean_accuracy_traditional_solver, mean_cost_traditional_solver[0,...],  xerr=std_accuracy_traditional_solver, yerr=std_cost_traditional_solver[0,...], fmt='s', color='C0')
    print(mean_accuracy_traditional_solver)
    # Exclude the zero-error reference before fitting a guide line in log space.
    log_err = np.log10(mean_accuracy_traditional_solver[1:])
    log_cost = np.log10(mean_cost_traditional_solver[0,...])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[0].loglog(10**x_fit, 10**y_fit, '--', color='C0', linewidth=2,
                #label=f'FVM ($\\varepsilon^{{{slope:.2f}}}$)')
                label=f'FVM')
    print("slope is ", slope)

    # axs[0].loglog(mean_accuracy_mpcno_solver, mean_cost_mpcno_solver[...,0], 'o-')
    print(mean_accuracy_mpcno_solver.shape, mean_cost_mpcno_solver.shape, std_accuracy_mpcno_solver.shape)
    
    axs[0].errorbar(mean_accuracy_mpcno_solver[...,1], mean_cost_mpcno_solver[...,0], xerr=std_accuracy_mpcno_solver[...,1], fmt='o', color='C1')
    log_err = np.log10(mean_accuracy_mpcno_solver[...,1]) 
    log_cost = np.log10(mean_cost_mpcno_solver[...,0]) 
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[0].loglog(10**x_fit, 10**y_fit, '--', color='C1', linewidth=2,
                #label=f'Neural operator ($\\varepsilon^{{{slope:.2f}}}$)')
                label=f'Neural operator')
    
    axs[0].set_xlabel(r"Rel. $L^1$ error")
    axs[0].set_ylabel("Floating-point cost")
    axs[0].legend(loc='lower left')
    axs[0].set_ylim(bottom=1e10)
    
    xticks = [0.1, 0.15, 0.2, 0.25] 
    axs[0].xaxis.set_major_locator(FixedLocator(xticks))
    axs[0].xaxis.set_minor_locator(NullLocator())
    axs[0].xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    axs[0].xaxis.get_major_formatter().set_useOffset(False)
    axs[0].set_xlim(min(xticks)*0.9, max(xticks)*1.1)
    
    
    # axs[1].loglog(mean_accuracy_traditional_solver, mean_cost_traditional_solver[...,1], 'o-')

    
    axs[1].errorbar(mean_accuracy_traditional_solver, mean_cost_traditional_solver[1,...], xerr=std_accuracy_traditional_solver, yerr=std_cost_traditional_solver[1,...], fmt='s', color='C2')
    log_err = np.log10(mean_accuracy_traditional_solver[1:]) 
    log_cost = np.log10(mean_cost_traditional_solver[1,...])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color='C2', linewidth=2,
                label=f'FVM (CPU)')
    

    axs[1].errorbar(mean_accuracy_mpcno_solver[...,1], mean_cost_mpcno_solver[...,2], xerr=std_accuracy_mpcno_solver[...,1], yerr=std_cost_mpcno_solver[...,2], fmt='o', color='C1')
    log_err = np.log10(mean_accuracy_mpcno_solver[...,1])
    log_cost = np.log10(mean_cost_mpcno_solver[...,2])
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color='C1', linewidth=2,
                label=f'Neural operator (GPU)')

    axs[1].errorbar(mean_accuracy_mpcno_solver[...,1], mean_cost_mpcno_solver[...,1], xerr=std_accuracy_mpcno_solver[...,1], yerr=std_cost_mpcno_solver[...,1], fmt='o', color='C3')
    log_err = np.log10(mean_accuracy_mpcno_solver[...,1])
    log_cost = np.log10(mean_cost_mpcno_solver[...,1])
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color='C3', linewidth=2,
                label=f'Neural operator (CPU)')
    

    axs[1].set_xlabel(r"Rel. $L^1$ error")
    axs[1].set_ylabel("Runtime (s)")
    axs[1].legend(loc='lower right')

    axs[1].xaxis.set_major_locator(FixedLocator(xticks))
    axs[1].xaxis.set_minor_locator(NullLocator())
    axs[1].xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    axs[1].xaxis.get_major_formatter().set_useOffset(False)
    axs[1].set_xlim(min(xticks)*0.9, max(xticks)*1.1)
    
    

    fig.tight_layout()
    fig.savefig("figs/aerodynamics_cost_accuracy.png")    
        
if __name__ == "__main__":
    cost_accuracy_plot()
