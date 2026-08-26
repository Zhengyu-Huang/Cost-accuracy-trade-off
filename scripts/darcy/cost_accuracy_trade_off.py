"""Create the main Darcy-flow field map and cost--accuracy figure.

Run from ``scripts/darcy`` so the relative data and figure paths resolve.
The default driver requires raw sample ``../../data/darcy/darcy_data_09999.npy``
and the traditional/MNO prediction and cost archives under ``data/``. It writes
``figs/darcy_flow_map.png`` and ``figs/darcy_flow_cost_accuracy.png``.
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
    


                        


def visualize_data(visualize_prediction=False):
    """Plot one reference field and optionally its FEM/MNO predictions."""
        
    i = 9999
    data = np.load(f"../../data/darcy/darcy_data_{i:05d}.npy")
    kappa_data, u_data = data[:,:,0], data[:,:,1]
    vmin_u, vmax_u = u_data.min(), u_data.max()
    
    ngrid, _ = u_data.shape
    x, y = np.meshgrid(np.linspace(0,1,ngrid), np.linspace(0,1,ngrid), indexing='ij')

    


    fig, axs = plt.subplots(1, 4, figsize=(24, 6)) if visualize_prediction else plt.subplots(1, 2, figsize=(16, 6))
    im = axs[0].pcolormesh(x, y, kappa_data, cmap='viridis', shading='gouraud')
    axs[0].set_title(r"$a$");
    axs[0].set_xticks([]);
    axs[0].set_yticks([]);
    axs[0].set_aspect('equal')
    cbar = fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04, shrink=1.0)
    cbar.ax.tick_params(axis='y', colors=tk)
    
    im = axs[1].pcolormesh(x, y, u_data, cmap='viridis', shading='gouraud', vmin=vmin_u, vmax=vmax_u)
    axs[1].set_title(r"$u~$(reference)");
    axs[1].set_xticks([]); 
    axs[1].set_yticks([]); 
    axs[1].set_aspect('equal')
    cbar = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04, shrink=1.0)
    cbar.ax.tick_params(axis='y', colors=tk)
    cbar.formatter = formatter

    if visualize_prediction:
        
        traditional_solver_pred = np.load('data/traditional_solver_data.npz')["sol"]
        mno_solver_pred = np.load('data/mno_solver_data.npz')["sol"]  

        print(traditional_solver_pred.shape)
        print(mno_solver_pred.shape)

        # Striding 513 nodal values by 8 gives a 64-by-64-cell FEM grid.
        stride = 2**3
        im = axs[2].pcolormesh(x[0::stride,0::stride], y[0::stride,0::stride], traditional_solver_pred, cmap='viridis', shading='gouraud', vmin=vmin_u, vmax=vmax_u)
        axs[2].set_title(r"$u~(64\times64)$");
        axs[2].set_xticks([]); 
        axs[2].set_yticks([]); 
        axs[2].set_aspect('equal')
        cbar = fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04, shrink=1.0)
        cbar.ax.tick_params(axis='y', colors=tk)
        cbar.formatter = formatter

        # The saved MNO prediction uses every fourth reference-grid node.
        stride = 2**2
        im = axs[3].pcolormesh(x[0::stride,0::stride], y[0::stride,0::stride], mno_solver_pred, cmap='viridis', shading='gouraud', vmin=vmin_u, vmax=vmax_u)
        axs[3].set_title(fr"$u~$(NO)");
        axs[3].set_xticks([]); 
        axs[3].set_yticks([]); 
        axs[3].set_aspect('equal')
        cbar = fig.colorbar(im, ax=axs[3], fraction=0.046, pad=0.04, shrink=1.0)
        cbar.ax.tick_params(axis='y', colors=tk)
        cbar.formatter = formatter

    fig.savefig(f"figs/darcy_flow_map.png")
    
    
    
def solution_plot():
    """Plot representative FEM solutions and the standalone trade-off curve."""
    # Pickle is required because solutions at different grids form an object array.
    cost_accuracy_traditional_solver_data = np.load('data/cost_accuracy_traditional_solver_data.npz', allow_pickle=True)
    cost = cost_accuracy_traditional_solver_data['cost']
    accuracy = cost_accuracy_traditional_solver_data['accuracy']
    sol = cost_accuracy_traditional_solver_data['sol'].tolist()   # entries stack [kappa, reference u, predicted u]

    ngrid, _, _ = sol[0].shape
    
    fig, axs = plt.subplots(2, 4, figsize=(16, 6))
    x, y = np.meshgrid(np.linspace(0,1,ngrid), np.linspace(0,1,ngrid), indexing='ij')
    im = axs[0,0].pcolormesh(x, y, sol[0][...,1], shading = "gouraud") # reference solution
    fig.colorbar(im, ax=axs[0,0])
    axs[0,0].set_title(fr'$u ({ngrid-1} \times {ngrid-1})$')
    im = axs[1,0].pcolormesh(x, y, sol[0][...,0], shading = "gouraud") # permeability
    fig.colorbar(im, ax=axs[1,0])
    axs[1,0].set_title(r'$\kappa$')
    
    formatter = ScalarFormatter(useMathText=True)
    # Keep small pointwise errors readable with a shared scientific exponent.
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    formatter.set_useOffset(True)
    
    for downsample in range(1,4):
        stride = 2**downsample
        im = axs[0,downsample].pcolormesh(x[0::stride, 0::stride], y[0::stride, 0::stride], sol[downsample][...,2], shading = "gouraud") # predicted solution
        fig.colorbar(im, ax=axs[0,downsample])
        axs[0,downsample].set_title(fr'$u ({(ngrid-1)//stride} \times {(ngrid-1)//stride})$')
        im = axs[1,downsample].pcolormesh(x[0::stride, 0::stride], y[0::stride, 0::stride], np.fabs(sol[downsample][...,2] - sol[downsample][...,1]), shading = "gouraud") # pointwise absolute error
        cbar = fig.colorbar(im, ax=axs[1,downsample])
        cbar.formatter = formatter
        cbar.update_ticks()
        axs[1,downsample].set_title('Error')
    fig.tight_layout()
    fig.savefig("figs/solution_traditional_solver.pdf")    
    
    
    
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Axis 1 contains repeated samples at each grid resolution.
    mean_cost = np.mean(cost, axis=1)     
    mean_accuracy = np.mean(accuracy, axis=1)    
    std_cost  = np.std(cost, axis=1, ddof=1)       
    std_accuracy  = np.std(accuracy, axis=1, ddof=1)

    axs[0].loglog(mean_accuracy, mean_cost[...,0], 'o-')
    axs[0].errorbar(mean_accuracy, mean_cost[...,0], xerr=std_accuracy, fmt='o')
    axs[0].set_xlabel("Rel. error")
    axs[0].set_ylabel("Floating-point cost")
    axs[1].loglog(mean_accuracy, mean_cost[...,1], 'o-')
    axs[1].errorbar(mean_accuracy, mean_cost[...,1], xerr=std_accuracy, yerr=std_cost[...,1], fmt='o')
    axs[1].set_xlabel("Rel. error")
    axs[1].set_ylabel("CPU cost (s)")

    fig.tight_layout()
    fig.savefig("figs/cost_accuracy_traditional_solver.pdf")    
        

def cost_accuracy_plot():
    """Compare FEM and MNO floating-point/runtime cost against relative error."""
    cost_accuracy_traditional_solver_data = np.load('data/cost_accuracy_traditional_solver_data.npz', allow_pickle=True)
    cost_traditional_solver = cost_accuracy_traditional_solver_data['cost']
    accuracy_traditional_solver = cost_accuracy_traditional_solver_data['accuracy']
    
    cost_accuracy_mno_solver_data = np.load('data/cost_accuracy_mno_solver_data.npz', allow_pickle=True)
    cost_mno_solver = cost_accuracy_mno_solver_data['cost']
    # Collapse all model/grid configuration axes while retaining trial and cost axes.
    cost_mno_solver = cost_mno_solver.reshape((-1, cost_mno_solver.shape[-2], cost_mno_solver.shape[-1]))
    accuracy_mno_solver = cost_accuracy_mno_solver_data['accuracy']
    accuracy_mno_solver = accuracy_mno_solver.reshape((-1, accuracy_mno_solver.shape[-1]))
    
        
    
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    for ax in axs:
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.6)
        
    # Cost channels are [FLOP estimate, CPU seconds] for FEM and
    # [FLOP estimate, CPU seconds, GPU seconds] for the MNO.
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
    
    axs[1].set_xlabel(r"Rel. $L^2$ error")
    axs[1].set_ylabel("Runtime (s)")
    axs[1].legend(loc='lower left')
    

    fig.tight_layout()
    fig.savefig("figs/darcy_flow_cost_accuracy.png")    
        
if __name__ == "__main__":
    visualize_data(visualize_prediction=True)
    cost_accuracy_plot()
