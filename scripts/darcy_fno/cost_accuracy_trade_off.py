"""Create the supplementary FEM/MNO/FNO Darcy cost--accuracy figure.

Run from ``scripts/darcy_fno``. The default driver reads the FEM and MNO
archives from ``../darcy/data`` and the FNO archive from local ``data/``. It
writes ``figs/supp_darcy_flow_cost_accuracy.png``.
Example: ``python3 cost_accuracy_trade_off.py``.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

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
# Use scientific notation with the exponent displayed as an axis offset.
formatter.set_scientific(True)
formatter.set_powerlimits((-2, 2))   # switch when the exponent reaches -2 or 2
formatter.set_useOffset(True)
lbl = "#000000"
tk = "#808080"
    




def cost_accuracy_plot():
    """Compare FEM, MNO, and FNO cost/error archives on shared axes."""
    cost_accuracy_traditional_solver_data = np.load('../darcy/data/cost_accuracy_traditional_solver_data.npz', allow_pickle=True)
    cost_traditional_solver = cost_accuracy_traditional_solver_data['cost']
    accuracy_traditional_solver = cost_accuracy_traditional_solver_data['accuracy']
    
    cost_accuracy_mno_solver_data = np.load('../darcy/data/cost_accuracy_mno_solver_data.npz', allow_pickle=True)
    cost_mno_solver = cost_accuracy_mno_solver_data['cost']
    # Collapse model/grid configuration axes while retaining trial and cost axes.
    cost_mno_solver = cost_mno_solver.reshape((-1, cost_mno_solver.shape[-2], cost_mno_solver.shape[-1]))
    accuracy_mno_solver = cost_accuracy_mno_solver_data['accuracy']
    accuracy_mno_solver = accuracy_mno_solver.reshape((-1, accuracy_mno_solver.shape[-1]))
    
        
    cost_accuracy_fno_solver_data = np.load('data/cost_accuracy_fno_solver_data.npz', allow_pickle=True)
    cost_fno_solver = cost_accuracy_fno_solver_data['cost']
    cost_fno_solver = cost_fno_solver.reshape((-1, cost_fno_solver.shape[-2], cost_fno_solver.shape[-1]))
    accuracy_fno_solver = cost_accuracy_fno_solver_data['accuracy']
    accuracy_fno_solver = accuracy_fno_solver.reshape((-1, accuracy_fno_solver.shape[-1]))
        
        
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    for ax in axs:
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.6)
        
    # FEM cost channels are [FLOP estimate, CPU seconds]; neural-operator cost
    # channels are [FLOP estimate, CPU seconds, GPU seconds].
    mean_cost_traditional_solver = np.mean(cost_traditional_solver, axis=1)     
    mean_accuracy_traditional_solver = np.mean(accuracy_traditional_solver, axis=1)    
    std_cost_traditional_solver  = np.std(cost_traditional_solver, axis=1, ddof=1)       
    std_accuracy_traditional_solver  = np.std(accuracy_traditional_solver, axis=1, ddof=1)
    
    
    
    mean_cost_mno_solver = np.mean(cost_mno_solver, axis=1)     
    mean_accuracy_mno_solver = np.mean(accuracy_mno_solver, axis=1)    
    std_cost_mno_solver  = np.std(cost_mno_solver, axis=1, ddof=1)       
    std_accuracy_mno_solver  = np.std(accuracy_mno_solver, axis=1, ddof=1)
    
    print("mean_cost_mno_solver: ", mean_cost_mno_solver)
    print("mean_accuracy_mno_solver:", mean_accuracy_mno_solver)
    
    
    mean_cost_fno_solver = np.mean(cost_fno_solver, axis=1)     
    mean_accuracy_fno_solver = np.mean(accuracy_fno_solver, axis=1)    
    std_cost_fno_solver  = np.std(cost_fno_solver, axis=1, ddof=1)       
    std_accuracy_fno_solver  = np.std(accuracy_fno_solver, axis=1, ddof=1)
    
    print("mean_cost_fno_solver: ", mean_cost_fno_solver)
    print("mean_accuracy_fno_solver:", mean_accuracy_fno_solver)

    # axs[0].loglog(mean_accuracy_traditional_solver, mean_cost_traditional_solver[...,0], 'o-')
    axs[0].errorbar(mean_accuracy_traditional_solver, mean_cost_traditional_solver[...,0], xerr=std_accuracy_traditional_solver, fmt='s', color='C0')
    # Fit power laws in log10 space; omit the finest-grid FEM point.
    log_err = np.log10(mean_accuracy_traditional_solver)[1:]
    log_cost = np.log10(mean_cost_traditional_solver[...,0])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[0].loglog(10**x_fit, 10**y_fit, '--', color='C0', linewidth=2,
                label=f'FEM ($\\varepsilon^{{{slope:.2f}}}$)')
        
    # axs[0].loglog(mean_accuracy_mno_solver, mean_cost_mno_solver[...,0], 'o-')
    axs[0].errorbar(mean_accuracy_mno_solver, mean_cost_mno_solver[...,0], xerr=std_accuracy_mno_solver, fmt='o', color='C1')
    log_err = np.log10(mean_accuracy_mno_solver) 
    log_cost = np.log10(mean_cost_mno_solver[...,0]) 
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[0].loglog(10**x_fit, 10**y_fit, '--', color='C1', linewidth=2,
                label=f'Neural operator ($\\varepsilon^{{{slope:.2f}}}$)')
    
    
    axs[0].errorbar(mean_accuracy_fno_solver, mean_cost_fno_solver[...,0], xerr=std_accuracy_fno_solver, fmt='d', color='C2')
    log_err = np.log10(mean_accuracy_fno_solver) 
    log_cost = np.log10(mean_cost_fno_solver[...,0]) 
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[0].loglog(10**x_fit, 10**y_fit, '--', color='C2', linewidth=2,
                label=f'FNO ($\\varepsilon^{{{slope:.2f}}}$)')
    
    
    
    axs[0].set_xlabel(r"Rel. $L^2$ error")
    axs[0].set_ylabel("Floating-point cost")
    axs[0].legend(loc='lower left')
    
    axs[0].set_ylim(bottom=1e4)
    
    # axs[1].loglog(mean_accuracy_traditional_solver, mean_cost_traditional_solver[...,1], 'o-')
    axs[1].errorbar(mean_accuracy_traditional_solver, mean_cost_traditional_solver[...,1], xerr=std_accuracy_traditional_solver, yerr=std_cost_traditional_solver[...,1], fmt='s', color='C0')
    log_err = np.log10(mean_accuracy_traditional_solver)[1:]
    log_cost = np.log10(mean_cost_traditional_solver[...,1])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color='C0', linewidth=2,
                label=f'FEM (CPU)')
    
    axs[1].errorbar(mean_accuracy_mno_solver, mean_cost_mno_solver[...,2], xerr=std_accuracy_mno_solver, yerr=std_cost_mno_solver[...,2], fmt='o', color='C1')
    log_err = np.log10(mean_accuracy_mno_solver)
    log_cost = np.log10(mean_cost_mno_solver[...,2])
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color='C1', linewidth=2,
                label=f'Neural operator (GPU)')

    axs[1].errorbar(mean_accuracy_mno_solver, mean_cost_mno_solver[...,1], xerr=std_accuracy_mno_solver, yerr=std_cost_mno_solver[...,1], fmt='o', color='C3')
    log_err = np.log10(mean_accuracy_mno_solver)
    log_cost = np.log10(mean_cost_mno_solver[...,1])
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color='C3', linewidth=2,
                label=f'Neural operator (CPU)')
    
    
    
    axs[1].errorbar(mean_accuracy_fno_solver, mean_cost_fno_solver[...,2], xerr=std_accuracy_fno_solver, yerr=std_cost_fno_solver[...,2], fmt='d', color='C2')
    log_err = np.log10(mean_accuracy_fno_solver)
    log_cost = np.log10(mean_cost_fno_solver[...,2])
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color='C2', linewidth=2,
                label=f'FNO (GPU)')

    axs[1].errorbar(mean_accuracy_fno_solver, mean_cost_fno_solver[...,1], xerr=std_accuracy_fno_solver, yerr=std_cost_fno_solver[...,1], fmt='d', color='C4')
    log_err = np.log10(mean_accuracy_fno_solver)
    log_cost = np.log10(mean_cost_fno_solver[...,1])
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color='C4', linewidth=2,
                label=f'FNO (CPU)')
        
        
    axs[1].set_xlabel(r"Rel. $L^2$ error")
    axs[1].set_ylabel("Runtime (s)")
    axs[1].legend(loc='lower left')
    

    fig.tight_layout()
    fig.savefig("figs/supp_darcy_flow_cost_accuracy.png")    
        
if __name__ == "__main__":
    cost_accuracy_plot()
