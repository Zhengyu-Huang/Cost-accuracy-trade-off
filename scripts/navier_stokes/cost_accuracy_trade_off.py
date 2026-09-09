import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
# plt.style.use('seaborn-v0_8-whitegrid')   # 现代网格样式
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
# 2. 强制使用科学计数法，并让指数作为偏移量（顶部显示）
formatter.set_scientific(True)
formatter.set_powerlimits((-2, 2))   # 数值小于 1e-3 或大于 1e3 时触发偏移量
formatter.set_useOffset(True)        # 明确使用偏移量    
lbl = "#000000"
tk = "#808080"
light_blue = "#ACD6ED"
dark_blue = "#0095FF"
light_green = '#98df8a'
dark_green = "#00FD00"
    
def visualize_data(visualize_prediction=False):
        
    i = 1999
    data = data = np.load(f"../../data/navier_stokes/navier_stokes_{i:05d}.npy")
    vorticity = data[1:, ...]
    force     = data[0, ...]
    
    ngrid, _ = force.shape
    x, y = np.meshgrid(np.linspace(0,1,ngrid, endpoint=False), np.linspace(0,1,ngrid, endpoint=False), indexing='ij')
    ts = [0,10,20,30]
    vmin_vorticity = [vorticity[ts[i],...].min() for i in range(len(ts))]
    vmax_vorticity = [vorticity[ts[i],...].max() for i in range(len(ts))]        
    fig, axs = plt.subplots(3, 4, figsize=(24, 18)) if visualize_prediction else plt.subplots(1, 4, figsize=(24, 6), squeeze=False)
    axs[0,0].set_ylabel(rf"$\omega~$(Reference)", fontsize=28)
    for i in range(4):
        im = axs[0,i].pcolormesh(x, y, vorticity[ts[i],...], cmap='viridis', shading='gouraud', vmin=vmin_vorticity[i], vmax=vmax_vorticity[i])
        axs[0,i].set_title(rf"$t={ts[i]}$");
        axs[0,i].set_xticks([]);
        axs[0,i].set_yticks([]);
        axs[0,i].set_aspect('equal')
        cbar = fig.colorbar(im, ax=axs[0,i], fraction=0.046, pad=0.04, shrink=1.0)
        cbar.ax.tick_params(axis='y', colors=tk)
        cbar.formatter = formatter

    if visualize_prediction:
        traditional_solver_pred = np.load('data/traditional_solver_data.npz')
        mno_solver_pred = np.load('data/mno_solver_data.npz')["sol"]  

        print(traditional_solver_pred.shape)
        print(mno_solver_pred.shape)

        stride = 2**2
        axs[1,0].set_ylabel(rf"$\omega~(64 \times 64)$", fontsize=28);
        for i in range(4):
            im = axs[1,i].pcolormesh(x[::stride,::stride], y[::stride,::stride], traditional_solver_pred[ts[i],...], cmap='viridis', shading='gouraud', vmin=vmin_vorticity[i], vmax=vmax_vorticity[i])
            axs[1,i].set_xticks([]);
            axs[1,i].set_yticks([]);
            axs[1,i].set_aspect('equal')
            cbar = fig.colorbar(im, ax=axs[1,i], fraction=0.046, pad=0.04, shrink=1.0)
            cbar.ax.tick_params(axis='y', colors=tk)
            cbar.formatter = formatter

        stride = 2**1
        axs[2,0].set_ylabel(rf"$\omega~$(Neural operator)", fontsize=28);
        for i in range(4):
            im = axs[2,i].pcolormesh(x[::stride,::stride], y[::stride,::stride], mno_solver_pred[ts[i],...], cmap='viridis', shading='gouraud', vmin=vmin_vorticity[i], vmax=vmax_vorticity[i])
            axs[2,i].set_xticks([]);
            axs[2,i].set_yticks([]);
            axs[2,i].set_aspect('equal')
            cbar = fig.colorbar(im, ax=axs[2,i], fraction=0.046, pad=0.04, shrink=1.0)
            cbar.ax.tick_params(axis='y', colors=tk)
            cbar.formatter = formatter
    

    fig.savefig(f"figs/navier_stokes_flow_map.png")
    
    
def solution_plot():
    cost_accuracy_traditional_solver_data = np.load('data/cost_accuracy_traditional_solver_data.npz', allow_pickle=True)   # 注意 allow_pickle=True
    cost = cost_accuracy_traditional_solver_data['cost']
    accuracy = cost_accuracy_traditional_solver_data['accuracy']
    sol = cost_accuracy_traditional_solver_data['sol'].tolist()   # list of [kappa_data, u_ref, u_data]

    ngrid, _, _ = sol[0].shape
    
    fig, axs = plt.subplots(2, 4, figsize=(32, 6))
    x, y = np.meshgrid(np.linspace(0,1,ngrid), np.linspace(0,1,ngrid), indexing='ij')
    im = axs[0,0].pcolormesh(x, y, sol[0][...,1], shading = "gouraud") # u_ref
    fig.colorbar(im, ax=axs[0,0])
    axs[0,0].set_title(fr'$u ({ngrid-1} \times {ngrid-1})$')
    im = axs[1,0].pcolormesh(x, y, sol[0][...,0], shading = "gouraud") # kappa_ref
    fig.colorbar(im, ax=axs[1,0])
    axs[1,0].set_title(r'$\kappa$')
    
    formatter = ScalarFormatter(useMathText=True)
    # 2. 强制使用科学计数法，并让指数作为偏移量（顶部显示）
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))   # 数值小于 1e-3 或大于 1e3 时触发偏移量
    formatter.set_useOffset(True)        # 明确使用偏移量
    
    for downsample in range(1,4):
        stride = 2**downsample
        im = axs[0,downsample].pcolormesh(x[0::stride, 0::stride], y[0::stride, 0::stride], sol[downsample][...,2], shading = "gouraud") # u_ref
        fig.colorbar(im, ax=axs[0,downsample])
        axs[0,downsample].set_title(fr'$u ({(ngrid-1)//stride} \times {(ngrid-1)//stride})$')
        im = axs[1,downsample].pcolormesh(x[0::stride, 0::stride], y[0::stride, 0::stride], np.fabs(sol[downsample][...,2] - sol[downsample][...,1]), shading = "gouraud") # kappa_ref
        cbar = fig.colorbar(im, ax=axs[1,downsample])
        cbar.formatter = formatter
        cbar.update_ticks()
        axs[1,downsample].set_title('Error')
    fig.tight_layout()
    fig.savefig("figs/solution_traditional_solver.pdf")    
    
    
    
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
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
    

def plot_with_dashed_errors(err):
    data_line, caplines, barlinecols = err.lines
    # Make the error-bar segments dashed.
    for bars in barlinecols:
        bars.set_linestyle(":")

def nrollouts_plot():
    
    cost_accuracy_traditional_solver_data = np.load('data/cost_accuracy_traditional_solver_data.npz', allow_pickle=True)   # 注意 allow_pickle=True
    cost_traditional_solver = cost_accuracy_traditional_solver_data['cost']
    # np.array of size (n_downsample, n_trial, 2, nt+1)
    accuracy_traditional_solver = cost_accuracy_traditional_solver_data['accuracy']
    
    # np.array of size ((len(nrollouts), n_trial, nt+1))
    accuracy_mno_solver = np.load('data/accuracy_mno_solver_nrollout_data.npz')['accuracy']
    
    fig, axs = plt.subplots(1, 1, figsize=(12, 6))
    axs.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
    
    mean_accuracy_traditional_solver = np.mean(accuracy_traditional_solver, axis=1)       
    std_accuracy_traditional_solver  = np.std(accuracy_traditional_solver, axis=1, ddof=1)
    mean_accuracy_mno_solver = np.mean(accuracy_mno_solver, axis=1)
    std_accuracy_mno_solver  = np.std(accuracy_mno_solver, axis=1, ddof=1)      
    
    nt = 50
    time_array = np.linspace(0, nt, nt+1)
    # axs[0].loglog(mean_accuracy_traditional_solver, mean_cost_traditional_solver[...,0], 'o-')
    # axs.errorbar(time_array, mean_accuracy_traditional_solver[0,0,...], yerr=std_accuracy_traditional_solver[0,0,...], fmt='-s', label=rf"Spectral method ($256 \times 256$)", color='C0')
    err = axs.errorbar(time_array, mean_accuracy_traditional_solver[1,0,...], yerr=std_accuracy_traditional_solver[1,0,...], fmt='-s', label=rf"Spectral method ($128 \times 128$)", color='C0')
    plot_with_dashed_errors(err)
    err = axs.errorbar(time_array, mean_accuracy_traditional_solver[2,0,...], yerr=std_accuracy_traditional_solver[2,0,...], fmt='-s', label=rf"Spectral method ($64 \times 64$)", color='C1')
    plot_with_dashed_errors(err)
    err = axs.errorbar(time_array, mean_accuracy_traditional_solver[3,0,...], yerr=std_accuracy_traditional_solver[3,0,...], fmt='-s', label=rf"Spectral method ($32 \times 32$)", color='C2')
    plot_with_dashed_errors(err)
    err = axs.errorbar(time_array, mean_accuracy_mno_solver[0,...], yerr=std_accuracy_mno_solver[0,...], fmt='-o', label=rf"Neural operator ($s=1$)", color='C3')
    plot_with_dashed_errors(err)
    err = axs.errorbar(time_array, mean_accuracy_mno_solver[1,...], yerr=std_accuracy_mno_solver[1,...], fmt='-o', label=rf"Neural operator ($s=2$)", color='C4')
    plot_with_dashed_errors(err)
    err = axs.errorbar(time_array, mean_accuracy_mno_solver[2,...], yerr=std_accuracy_mno_solver[2,...], fmt='-o', label=rf"Neural operator ($s=3$)", color='C5')
    plot_with_dashed_errors(err)
    

    axs.legend()
    axs.set_xlabel("Time")
    axs.set_ylabel(r"Rel. $L^2$ error")
    

    fig.tight_layout()
    fig.savefig("figs/navier_stokes_nrollout_accuracy.png")   
    


def set_accuracy_ticks(ax):
    major_ticks = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    major_labels = [
        r"$10^{-9}$", r"$10^{-8}$", r"$10^{-7}$", r"$10^{-6}$", r"$10^{-5}$",
        r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$",
    ]

    # 2e-4,...,9e-4；2e-3,...,9e-3；2e-2,...,9e-2
    minor_ticks = [
        multiplier * 10.0**exponent
        for exponent in (-4, -3, -2)
        for multiplier in range(2, 10)
    ]

    ax.set_xscale("log")
    ax.set_xticks(major_ticks)
    ax.set_xticklabels(major_labels)
    ax.set_xticks(minor_ticks, minor=True)
    ax.tick_params(axis="x", which="minor", labelbottom=False)

    ax.grid(axis="x", which="major", linestyle=":", linewidth=0.8)
    ax.grid(axis="x", which="minor", linestyle=":", linewidth=0.45, alpha=0.6)
    ax.set_xlim(2e-10, 2e-1)

def cost_accuracy_plot(nt_error=30, nt=50):
    """Plot cost--accuracy data through prediction horizon ``nt_error``.

    Accuracy is averaged over predicted states at times 1 through
    ``nt_error``; the exactly prescribed initial state at time 0 is excluded.
    """
    if not 1 <= nt_error <= nt:
        raise ValueError(f"nt_error must satisfy 1 <= nt_error <= {nt}")
    
    cost_accuracy_traditional_solver_data = np.load('data/cost_accuracy_traditional_solver_data.npz', allow_pickle=True)   # 注意 allow_pickle=True
    # np.array of size (n_downsample, n_trial, 2)
    cost_traditional_solver = cost_accuracy_traditional_solver_data['cost'] * nt_error / nt
    # np.array of size (n_downsample, n_trial, 2, nt+1) -> (n_downsample, n_trial, 2)
    accuracy_traditional_solver = np.mean(cost_accuracy_traditional_solver_data['accuracy'][...,1:nt_error+1], axis=3)
    
    cost_accuracy_mno_solver_data = np.load('data/cost_accuracy_mno_solver_data.npz')
    # np.array of size (len(downsample_values), len(k_max_values), len(n_layer_values), len(df_values), n_trial, 3)
    cost_mno_solver = cost_accuracy_mno_solver_data['cost']
    cost_mno_solver[...,0] *= nt_error  # we save only one step flops
    cost_mno_solver[...,1:] *= nt_error / nt


    print("cost_mno_solver = " , cost_mno_solver[0,0,:,0,0,:])
    print("cost_mno_solver = " , cost_mno_solver[1,0,:,0,0,:])
    print("cost_mno_solver = " , cost_mno_solver[2,0,:,0,0,:])


    cost_mno_solver = cost_mno_solver.reshape((-1, cost_mno_solver.shape[-2], cost_mno_solver.shape[-1]))
    
    # np.array of size (len(downsample_values), len(k_max_values), len(n_layer_values), len(df_values), n_trial, 2, nt+1) -> (len(downsample_values), len(k_max_values), len(n_layer_values), len(df_values), n_trial, 2)
    accuracy_mno_solver = np.mean(cost_accuracy_mno_solver_data['accuracy'][...,1:nt_error+1], axis=6)



    print("accuracy_mno_solver = ", accuracy_mno_solver[0,0,1,0,:,:])
    print("accuracy_mno_solver = ", accuracy_mno_solver[1,0,1,0,0,:])
    print("accuracy_mno_solver = ", accuracy_mno_solver[2,0,1,0,0,:])
    

    accuracy_mno_solver = accuracy_mno_solver.reshape((-1, accuracy_mno_solver.shape[-2], accuracy_mno_solver.shape[-1]))
        



    fig, axs = plt.subplots(1, 2, figsize=(17, 6))
    for ax in axs:
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.6)
    

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
    axs[0].errorbar(mean_accuracy_traditional_solver[...,0], mean_cost_traditional_solver[...,0], xerr=std_accuracy_traditional_solver[...,0], fmt='s', color='C0')
    log_err = np.log10(mean_accuracy_traditional_solver[...,0])[1:]
    log_cost = np.log10(mean_cost_traditional_solver[...,0])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[0].loglog(10**x_fit, 10**y_fit, '--', color='C0', linewidth=2,
                label=f'Spectral method ($\\varepsilon^{{{slope:.2f}}}$)')
        
    # axs[0].loglog(mean_accuracy_mno_solver, mean_cost_mno_solver[...,0], 'o-')
    print(mean_accuracy_mno_solver.shape, mean_cost_mno_solver.shape, std_accuracy_mno_solver.shape)
    
    axs[0].errorbar(mean_accuracy_mno_solver[...,0], mean_cost_mno_solver[...,0], xerr=std_accuracy_mno_solver[...,0], fmt='o', color='C1')
    log_err = np.log10(mean_accuracy_mno_solver[...,0]) 
    log_cost = np.log10(mean_cost_mno_solver[...,0]) 
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[0].loglog(10**x_fit, 10**y_fit, '--', color='C1', linewidth=2,
                label=f'Neural operator ($\\varepsilon^{{{slope:.2f}}}$)' if nt_error==1 else 'Neural operator')
    
    axs[0].set_xlabel(r"Rel. $L^2$ error")
    axs[0].set_ylabel("Floating-point cost")
    axs[0].legend(loc='lower left')
    axs[0].set_ylim(bottom=1e8 if nt_error == 30 else 1e7)

    set_accuracy_ticks(axs[0])
    # axs[1].loglog(mean_accuracy_traditional_solver, mean_cost_traditional_solver[...,1], 'o-')

    
    axs[1].errorbar(mean_accuracy_traditional_solver[...,0], mean_cost_traditional_solver[...,2], xerr=std_accuracy_traditional_solver[...,0], yerr=std_cost_traditional_solver[...,2], fmt='s', color='C2')
    log_err = np.log10(mean_accuracy_traditional_solver[...,0])[1:]
    log_cost = np.log10(mean_cost_traditional_solver[...,2])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color='C2', linewidth=2,
                label=f'Spectral method (GPU)')
    


    axs[1].errorbar(mean_accuracy_traditional_solver[...,0], mean_cost_traditional_solver[...,1], xerr=std_accuracy_traditional_solver[...,0], yerr=std_cost_traditional_solver[...,1], fmt='s', color='C0')
    log_err = np.log10(mean_accuracy_traditional_solver[...,0])[1:]
    log_cost = np.log10(mean_cost_traditional_solver[...,1])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color='C0', linewidth=2,
                label=f'Spectral method (CPU)')
    

    axs[1].errorbar(mean_accuracy_mno_solver[...,0], mean_cost_mno_solver[...,2], xerr=std_accuracy_mno_solver[...,0], yerr=std_cost_mno_solver[...,2], fmt='o', color='C1')
    log_err = np.log10(mean_accuracy_mno_solver[...,0])
    log_cost = np.log10(mean_cost_mno_solver[...,2])
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color='C1', linewidth=2,
                label=f'Neural operator (GPU)')

    axs[1].errorbar(mean_accuracy_mno_solver[...,0], mean_cost_mno_solver[...,1], xerr=std_accuracy_mno_solver[...,0], yerr=std_cost_mno_solver[...,1], fmt='o', color='C3')
    log_err = np.log10(mean_accuracy_mno_solver[...,0])
    log_cost = np.log10(mean_cost_mno_solver[...,1])
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color='C3', linewidth=2,
                label=f'Neural operator (CPU)')
    set_accuracy_ticks(axs[1])

    
    axs[1].set_xlabel(r"Rel. $L^2$ error")
    axs[1].set_ylabel("Runtime (s)")
    axs[1].legend(loc='lower left')
    

    
    fig.suptitle(f'T={nt_error}', y=0.92) 
    fig.tight_layout()
    fig.savefig(f"figs/navier_stokes_cost_accuracy_{nt_error}.png")    


def cost_accuracy_onestep_plot():
    """Plot cost--accuracy data through prediction horizon ``nt_error``.

    Accuracy is averaged over predicted states at times 1 through
    ``nt_error``; the exactly prescribed initial state at time 0 is excluded.
    """
    nt_error=1

    cost_accuracy_traditional_solver_data = np.load('data/cost_accuracy_traditional_solver_onestep_data.npz', allow_pickle=True)   # 注意 allow_pickle=True
    # np.array of size (n_downsample, n_trial, 2)
    cost_traditional_solver = cost_accuracy_traditional_solver_data['cost']
    # np.array of size (n_downsample, n_trial, 2, nt+1) -> (n_downsample, n_trial, 2)
    accuracy_traditional_solver = np.mean(cost_accuracy_traditional_solver_data['accuracy'][...,:], axis=3)
    
    cost_accuracy_mno_solver_data = np.load('data/cost_accuracy_mno_solver_onestep_data.npz')
    # np.array of size (len(downsample_values), len(k_max_values), len(n_layer_values), len(df_values), n_trial, 3)
    cost_mno_solver = cost_accuracy_mno_solver_data['cost']
    

    print("cost_mno_solver = " , cost_mno_solver[0,0,:,0,0,:])
    print("cost_mno_solver = " , cost_mno_solver[1,0,:,0,0,:])
    print("cost_mno_solver = " , cost_mno_solver[2,0,:,0,0,:])


    cost_mno_solver = cost_mno_solver.reshape((-1, cost_mno_solver.shape[-2], cost_mno_solver.shape[-1]))
    
    # np.array of size (len(downsample_values), len(k_max_values), len(n_layer_values), len(df_values), n_trial, 2, nt+1) -> (len(downsample_values), len(k_max_values), len(n_layer_values), len(df_values), n_trial, 2)
    accuracy_mno_solver = np.mean(cost_accuracy_mno_solver_data['accuracy'][...,:], axis=6)



    print("accuracy_mno_solver = ", accuracy_mno_solver[0,0,1,0,:,:])
    print("accuracy_mno_solver = ", accuracy_mno_solver[1,0,1,0,0,:])
    print("accuracy_mno_solver = ", accuracy_mno_solver[2,0,1,0,0,:])
    

    accuracy_mno_solver = accuracy_mno_solver.reshape((-1, accuracy_mno_solver.shape[-2], accuracy_mno_solver.shape[-1]))
        



    fig, axs = plt.subplots(1, 2, figsize=(17, 6))
    for ax in axs:
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.6)
    

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
    axs[0].errorbar(mean_accuracy_traditional_solver[...,0], mean_cost_traditional_solver[...,0], xerr=std_accuracy_traditional_solver[...,0], fmt='s', color='C0')
    log_err = np.log10(mean_accuracy_traditional_solver[...,0])[1:]
    log_cost = np.log10(mean_cost_traditional_solver[...,0])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[0].loglog(10**x_fit, 10**y_fit, '--', color='C0', linewidth=2,
                label=f'Spectral method ($\\varepsilon^{{{slope:.2f}}}$)')
        
    # axs[0].loglog(mean_accuracy_mno_solver, mean_cost_mno_solver[...,0], 'o-')
    print(mean_accuracy_mno_solver.shape, mean_cost_mno_solver.shape, std_accuracy_mno_solver.shape)
    
    axs[0].errorbar(mean_accuracy_mno_solver[...,0], mean_cost_mno_solver[...,0], xerr=std_accuracy_mno_solver[...,0], fmt='o', color='C1')
    log_err = np.log10(mean_accuracy_mno_solver[...,0]) 
    log_cost = np.log10(mean_cost_mno_solver[...,0]) 
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[0].loglog(10**x_fit, 10**y_fit, '--', color='C1', linewidth=2,
                label=f'Neural operator ($\\varepsilon^{{{slope:.2f}}}$)' if nt_error==1 else 'Neural operator')
    
    axs[0].set_xlabel(r"Rel. $L^2$ error")
    axs[0].set_ylabel("Floating-point cost")
    axs[0].legend(loc='lower left')
    axs[0].set_ylim(bottom=1e8 if nt_error == 30 else 1e7)
    set_accuracy_ticks(axs[0])
    # axs[1].loglog(mean_accuracy_traditional_solver, mean_cost_traditional_solver[...,1], 'o-')

    
    axs[1].errorbar(mean_accuracy_traditional_solver[...,0], mean_cost_traditional_solver[...,2], xerr=std_accuracy_traditional_solver[...,0], yerr=std_cost_traditional_solver[...,2], fmt='s', color='C2')
    log_err = np.log10(mean_accuracy_traditional_solver[...,0])[1:]
    log_cost = np.log10(mean_cost_traditional_solver[...,2])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color='C2', linewidth=2,
                label=f'Spectral method (GPU)')
    


    axs[1].errorbar(mean_accuracy_traditional_solver[...,0], mean_cost_traditional_solver[...,1], xerr=std_accuracy_traditional_solver[...,0], yerr=std_cost_traditional_solver[...,1], fmt='s', color='C0')
    log_err = np.log10(mean_accuracy_traditional_solver[...,0])[1:]
    log_cost = np.log10(mean_cost_traditional_solver[...,1])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color='C0', linewidth=2,
                label=f'Spectral method (CPU)')
    

    axs[1].errorbar(mean_accuracy_mno_solver[...,0], mean_cost_mno_solver[...,2], xerr=std_accuracy_mno_solver[...,0], yerr=std_cost_mno_solver[...,2], fmt='o', color='C1')
    log_err = np.log10(mean_accuracy_mno_solver[...,0])
    log_cost = np.log10(mean_cost_mno_solver[...,2])
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color='C1', linewidth=2,
                label=f'Neural operator (GPU)')

    axs[1].errorbar(mean_accuracy_mno_solver[...,0], mean_cost_mno_solver[...,1], xerr=std_accuracy_mno_solver[...,0], yerr=std_cost_mno_solver[...,1], fmt='o', color='C3')
    log_err = np.log10(mean_accuracy_mno_solver[...,0])
    log_cost = np.log10(mean_cost_mno_solver[...,1])
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color='C3', linewidth=2,
                label=f'Neural operator (CPU)')
    set_accuracy_ticks(axs[1])


    
    axs[1].set_xlabel(r"Rel. $L^2$ error")
    axs[1].set_ylabel("Runtime (s)")
    axs[1].legend(loc='lower left')
    

    
    fig.suptitle(f'T={nt_error}', y=0.92) 
    fig.tight_layout()
    fig.savefig(f"figs/navier_stokes_cost_accuracy_{nt_error}.png")    


def cost_accuracy_plot_etd(nt_error=30, nt=50):
    """Plot cost--accuracy data through prediction horizon ``nt_error``.

    Accuracy is averaged over predicted states at times 1 through
    ``nt_error``; the exactly prescribed initial state at time 0 is excluded.
    """
    if not 1 <= nt_error <= nt:
        raise ValueError(f"nt_error must satisfy 1 <= nt_error <= {nt}")
    
    cost_accuracy_traditional_solver_data = np.load('data/cost_accuracy_traditional_solver_data.npz', allow_pickle=True)   # 注意 allow_pickle=True
    # np.array of size (n_downsample, n_trial, 2)
    cost_traditional_solver = cost_accuracy_traditional_solver_data['cost'] * nt_error / nt
    # np.array of size (n_downsample, n_trial, 2, nt+1) -> (n_downsample, n_trial, 2)
    accuracy_traditional_solver = np.mean(cost_accuracy_traditional_solver_data['accuracy'][...,1:nt_error+1], axis=3)
    
    cost_accuracy_traditional_solver_data_etd1 = np.load('data/cost_accuracy_traditional_solver_data_ETD1.npz', allow_pickle=True)   # 注意 allow_pickle=True
    # np.array of size (n_downsample, n_trial, 2)
    cost_traditional_solver_etd1 = cost_accuracy_traditional_solver_data_etd1['cost'] * nt_error / nt
    # np.array of size (n_downsample, n_trial, 2, nt+1) -> (n_downsample, n_trial, 2)
    accuracy_traditional_solver_etd1 = np.mean(cost_accuracy_traditional_solver_data_etd1['accuracy'][...,1:nt_error+1], axis=3)
        
    cost_accuracy_traditional_solver_data_etdrk4 = np.load('data/cost_accuracy_traditional_solver_data_ETDRK4.npz', allow_pickle=True)   # 注意 allow_pickle=True
    # np.array of size (n_downsample, n_trial, 2)
    cost_traditional_solver_etdrk4 = cost_accuracy_traditional_solver_data_etdrk4['cost'] * nt_error / nt
    # np.array of size (n_downsample, n_trial, 2, nt+1) -> (n_downsample, n_trial, 2)
    accuracy_traditional_solver_etdrk4 = np.mean(cost_accuracy_traditional_solver_data_etdrk4['accuracy'][...,1:nt_error+1], axis=3)
        
    


    cost_accuracy_mno_solver_data = np.load('data/cost_accuracy_mno_solver_data.npz')
    # np.array of size (len(downsample_values), len(k_max_values), len(n_layer_values), len(df_values), n_trial, 3)
    cost_mno_solver = cost_accuracy_mno_solver_data['cost']
    cost_mno_solver[...,0] *= nt_error  # we save only one step flops
    cost_mno_solver[...,1:] *= nt_error / nt


    print("cost_mno_solver = " , cost_mno_solver[0,0,:,0,0,:])
    print("cost_mno_solver = " , cost_mno_solver[1,0,:,0,0,:])
    print("cost_mno_solver = " , cost_mno_solver[2,0,:,0,0,:])


    cost_mno_solver = cost_mno_solver.reshape((-1, cost_mno_solver.shape[-2], cost_mno_solver.shape[-1]))
    
    # np.array of size (len(downsample_values), len(k_max_values), len(n_layer_values), len(df_values), n_trial, 2, nt+1) -> (len(downsample_values), len(k_max_values), len(n_layer_values), len(df_values), n_trial, 2)
    accuracy_mno_solver = np.mean(cost_accuracy_mno_solver_data['accuracy'][...,1:nt_error+1], axis=6)



    print("accuracy_mno_solver = ", accuracy_mno_solver[0,0,1,0,:,:])
    print("accuracy_mno_solver = ", accuracy_mno_solver[1,0,1,0,0,:])
    print("accuracy_mno_solver = ", accuracy_mno_solver[2,0,1,0,0,:])
    

    accuracy_mno_solver = accuracy_mno_solver.reshape((-1, accuracy_mno_solver.shape[-2], accuracy_mno_solver.shape[-1]))
        



    fig, axs = plt.subplots(1, 2, figsize=(17, 6))
    for ax in axs:
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.6)
    
    mean_cost_traditional_solver = np.mean(cost_traditional_solver, axis=1) 
    mean_accuracy_traditional_solver = np.mean(accuracy_traditional_solver, axis=1)    
    std_cost_traditional_solver  = np.std(cost_traditional_solver, axis=1, ddof=1)       
    std_accuracy_traditional_solver  = np.std(accuracy_traditional_solver, axis=1, ddof=1)

    mean_cost_traditional_solver_etd1 = np.mean(cost_traditional_solver_etd1, axis=1) 
    mean_accuracy_traditional_solver_etd1 = np.mean(accuracy_traditional_solver_etd1, axis=1)    
    std_cost_traditional_solver_etd1  = np.std(cost_traditional_solver_etd1, axis=1, ddof=1)       
    std_accuracy_traditional_solver_etd1  = np.std(accuracy_traditional_solver_etd1, axis=1, ddof=1)

    mean_cost_traditional_solver_etdrk4 = np.mean(cost_traditional_solver_etdrk4, axis=1) 
    mean_accuracy_traditional_solver_etdrk4 = np.mean(accuracy_traditional_solver_etdrk4, axis=1)    
    std_cost_traditional_solver_etdrk4  = np.std(cost_traditional_solver_etdrk4, axis=1, ddof=1)       
    std_accuracy_traditional_solver_etdrk4  = np.std(accuracy_traditional_solver_etdrk4, axis=1, ddof=1)


    mean_cost_mno_solver = np.mean(cost_mno_solver, axis=1)  
    mean_accuracy_mno_solver = np.mean(accuracy_mno_solver, axis=1)    
    std_cost_mno_solver  = np.std(cost_mno_solver, axis=1, ddof=1)       
    std_accuracy_mno_solver  = np.std(accuracy_mno_solver, axis=1, ddof=1)

    print("mean_cost_mno_solver: ", mean_cost_mno_solver)
    print("mean_accuracy_mno_solver:", mean_accuracy_mno_solver)

    # axs[0].loglog(mean_accuracy_traditional_solver, mean_cost_traditional_solver[...,0], 'o-')
    axs[0].errorbar(mean_accuracy_traditional_solver[...,0], mean_cost_traditional_solver[...,0], xerr=std_accuracy_traditional_solver[...,0], fmt='s', color='C0')
    log_err = np.log10(mean_accuracy_traditional_solver[...,0])[1:]
    log_cost = np.log10(mean_cost_traditional_solver[...,0])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[0].loglog(10**x_fit, 10**y_fit, '--', color='C0', linewidth=2,
                label=f'RK4 ($\\varepsilon^{{{slope:.2f}}}$)')
        
    # axs[0].loglog(mean_accuracy_mno_solver, mean_cost_mno_solver[...,0], 'o-')
    print(mean_accuracy_mno_solver.shape, mean_cost_mno_solver.shape, std_accuracy_mno_solver.shape)

    axs[0].errorbar(mean_accuracy_traditional_solver_etd1[...,0], mean_cost_traditional_solver_etd1[...,0], xerr=std_accuracy_traditional_solver_etd1[...,0], fmt='p', color=light_blue)
    log_err = np.log10(mean_accuracy_traditional_solver_etd1[...,0])[1:]
    log_cost = np.log10(mean_cost_traditional_solver_etd1[...,0])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[0].loglog(10**x_fit, 10**y_fit, '--', color=light_blue, linewidth=2,
                label=f'ETD1 ($\\varepsilon^{{{slope:.2f}}}$)')
                # label=f'Spectral method ETD1 ($\\varepsilon^{{{slope:.2f}}}$)')

    axs[0].errorbar(mean_accuracy_traditional_solver_etdrk4[...,0], mean_cost_traditional_solver_etdrk4[...,0], xerr=std_accuracy_traditional_solver_etdrk4[...,0], fmt='x', color=dark_blue)
    log_err = np.log10(mean_accuracy_traditional_solver_etdrk4[...,0])[1:]
    log_cost = np.log10(mean_cost_traditional_solver_etdrk4[...,0])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[0].loglog(10**x_fit, 10**y_fit, '--', color=dark_blue, linewidth=2,
                label=f'ETDRK4 ($\\varepsilon^{{{slope:.2f}}}$)')
                # label=f'Spectral method etdrk4 ($\\varepsilon^{{{slope:.2f}}}$)')






    axs[0].errorbar(mean_accuracy_mno_solver[...,0], mean_cost_mno_solver[...,0], xerr=std_accuracy_mno_solver[...,0], fmt='o', color='C1')
    log_err = np.log10(mean_accuracy_mno_solver[...,0]) 
    log_cost = np.log10(mean_cost_mno_solver[...,0]) 
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[0].loglog(10**x_fit, 10**y_fit, '--', color='C1', linewidth=2,
                label=f'NO ($\\varepsilon^{{{slope:.2f}}}$)' if nt_error==1 else 'NO')
    
    axs[0].set_xlabel(r"Rel. $L^2$ error")
    axs[0].set_ylabel("Floating-point cost")
    axs[0].legend(loc='lower left')
    axs[0].set_ylim(bottom=1e8 if nt_error == 30 else 1e7)
    # axs[1].loglog(mean_accuracy_traditional_solver, mean_cost_traditional_solver[...,1], 'o-')
    set_accuracy_ticks(axs[0])
    
    axs[1].errorbar(mean_accuracy_traditional_solver[...,0], mean_cost_traditional_solver[...,2], xerr=std_accuracy_traditional_solver[...,0], yerr=std_cost_traditional_solver[...,2], fmt='s', color='C2')
    log_err = np.log10(mean_accuracy_traditional_solver[...,0])[1:]
    log_cost = np.log10(mean_cost_traditional_solver[...,2])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color='C2', linewidth=2,
                label=f'RK4 (GPU)')
    


    axs[1].errorbar(mean_accuracy_traditional_solver[...,0], mean_cost_traditional_solver[...,1], xerr=std_accuracy_traditional_solver[...,0], yerr=std_cost_traditional_solver[...,1], fmt='s', color='C0')
    log_err = np.log10(mean_accuracy_traditional_solver[...,0])[1:]
    log_cost = np.log10(mean_cost_traditional_solver[...,1])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color='C0', linewidth=2,
                label=f'RK4 (CPU)')




    axs[1].errorbar(mean_accuracy_traditional_solver_etd1[...,0], mean_cost_traditional_solver_etd1[...,2], xerr=std_accuracy_traditional_solver_etd1[...,0], yerr=std_cost_traditional_solver_etd1[...,2], fmt='p', color=light_green)
    log_err = np.log10(mean_accuracy_traditional_solver_etd1[...,0])[1:]
    log_cost = np.log10(mean_cost_traditional_solver_etd1[...,2])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color=light_green, linewidth=2,
                label=f'ETD1 (GPU)')
    


    axs[1].errorbar(mean_accuracy_traditional_solver_etd1[...,0], mean_cost_traditional_solver_etd1[...,1], xerr=std_accuracy_traditional_solver_etd1[...,0], yerr=std_cost_traditional_solver_etd1[...,1], fmt='p', color=light_blue)
    log_err = np.log10(mean_accuracy_traditional_solver_etd1[...,0])[1:]
    log_cost = np.log10(mean_cost_traditional_solver_etd1[...,1])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color=light_blue, linewidth=2,
                label=f'ETD1 (CPU)')



    axs[1].errorbar(mean_accuracy_traditional_solver_etdrk4[...,0], mean_cost_traditional_solver_etdrk4[...,2], xerr=std_accuracy_traditional_solver_etdrk4[...,0], yerr=std_cost_traditional_solver_etdrk4[...,2], fmt='x', color=dark_green)
    log_err = np.log10(mean_accuracy_traditional_solver_etdrk4[...,0])[1:]
    log_cost = np.log10(mean_cost_traditional_solver_etdrk4[...,2])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color=dark_green, linewidth=2,
                label=f'ETDRK4 (GPU)')
    


    axs[1].errorbar(mean_accuracy_traditional_solver_etdrk4[...,0], mean_cost_traditional_solver_etdrk4[...,1], xerr=std_accuracy_traditional_solver_etdrk4[...,0], yerr=std_cost_traditional_solver_etdrk4[...,1], fmt='x', color=dark_blue)
    log_err = np.log10(mean_accuracy_traditional_solver_etdrk4[...,0])[1:]
    log_cost = np.log10(mean_cost_traditional_solver_etdrk4[...,1])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color=dark_blue, linewidth=2,
                label=f'ETDRK4 (CPU)')




    axs[1].errorbar(mean_accuracy_mno_solver[...,0], mean_cost_mno_solver[...,2], xerr=std_accuracy_mno_solver[...,0], yerr=std_cost_mno_solver[...,2], fmt='o', color='C1')
    log_err = np.log10(mean_accuracy_mno_solver[...,0])
    log_cost = np.log10(mean_cost_mno_solver[...,2])
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color='C1', linewidth=2,
                label=f'NO (GPU)')

    axs[1].errorbar(mean_accuracy_mno_solver[...,0], mean_cost_mno_solver[...,1], xerr=std_accuracy_mno_solver[...,0], yerr=std_cost_mno_solver[...,1], fmt='o', color='C3')
    log_err = np.log10(mean_accuracy_mno_solver[...,0])
    log_cost = np.log10(mean_cost_mno_solver[...,1])
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color='C3', linewidth=2,
                label=f'NO (CPU)')
    
    set_accuracy_ticks(axs[1])

    
    axs[1].set_xlabel(r"Rel. $L^2$ error")
    axs[1].set_ylabel("Runtime (s)")
    axs[1].legend(loc='lower left')
    

    
    fig.suptitle(f'T={nt_error}', y=0.92) 
    fig.tight_layout()
    fig.savefig(f"figs/navier_stokes_cost_accuracy_{nt_error}.png")    


def cost_accuracy_onestep_plot_etd():
    """Plot cost--accuracy data through prediction horizon ``nt_error``.

    Accuracy is averaged over predicted states at times 1 through
    ``nt_error``; the exactly prescribed initial state at time 0 is excluded.
    """
    nt_error=1

    cost_accuracy_traditional_solver_data = np.load('data/cost_accuracy_traditional_solver_onestep_data.npz', allow_pickle=True)   # 注意 allow_pickle=True
    # np.array of size (n_downsample, n_trial, 2)
    cost_traditional_solver = cost_accuracy_traditional_solver_data['cost']
    # np.array of size (n_downsample, n_trial, 2, nt+1) -> (n_downsample, n_trial, 2)
    accuracy_traditional_solver = np.mean(cost_accuracy_traditional_solver_data['accuracy'][...,:], axis=3)

    cost_accuracy_traditional_solver_data_etd1 = np.load('data/cost_accuracy_traditional_solver_onestep_data_ETD1.npz', allow_pickle=True)   # 注意 allow_pickle=True
    # np.array of size (n_downsample, n_trial, 2)
    cost_traditional_solver_etd1 = cost_accuracy_traditional_solver_data_etd1['cost']
    # np.array of size (n_downsample, n_trial, 2, nt+1) -> (n_downsample, n_trial, 2)
    accuracy_traditional_solver_etd1 = np.mean(cost_accuracy_traditional_solver_data_etd1['accuracy'][...,:], axis=3)

    cost_accuracy_traditional_solver_data_etdrk4 = np.load('data/cost_accuracy_traditional_solver_onestep_data_ETDRK4.npz', allow_pickle=True)   # 注意 allow_pickle=True
    # np.array of size (n_downsample, n_trial, 2)
    cost_traditional_solver_etdrk4 = cost_accuracy_traditional_solver_data_etdrk4['cost']
    # np.array of size (n_downsample, n_trial, 2, nt+1) -> (n_downsample, n_trial, 2)
    accuracy_traditional_solver_etdrk4 = np.mean(cost_accuracy_traditional_solver_data_etdrk4['accuracy'][...,:], axis=3)

    
    cost_accuracy_mno_solver_data = np.load('data/cost_accuracy_mno_solver_onestep_data.npz')
    # np.array of size (len(downsample_values), len(k_max_values), len(n_layer_values), len(df_values), n_trial, 3)
    cost_mno_solver = cost_accuracy_mno_solver_data['cost']
    

    print("cost_mno_solver = " , cost_mno_solver[0,0,:,0,0,:])
    print("cost_mno_solver = " , cost_mno_solver[1,0,:,0,0,:])
    print("cost_mno_solver = " , cost_mno_solver[2,0,:,0,0,:])


    cost_mno_solver = cost_mno_solver.reshape((-1, cost_mno_solver.shape[-2], cost_mno_solver.shape[-1]))
    
    # np.array of size (len(downsample_values), len(k_max_values), len(n_layer_values), len(df_values), n_trial, 2, nt+1) -> (len(downsample_values), len(k_max_values), len(n_layer_values), len(df_values), n_trial, 2)
    accuracy_mno_solver = np.mean(cost_accuracy_mno_solver_data['accuracy'][...,:], axis=6)



    print("accuracy_mno_solver = ", accuracy_mno_solver[0,0,1,0,:,:])
    print("accuracy_mno_solver = ", accuracy_mno_solver[1,0,1,0,0,:])
    print("accuracy_mno_solver = ", accuracy_mno_solver[2,0,1,0,0,:])
    

    accuracy_mno_solver = accuracy_mno_solver.reshape((-1, accuracy_mno_solver.shape[-2], accuracy_mno_solver.shape[-1]))
        



    fig, axs = plt.subplots(1, 2, figsize=(17, 6))
    for ax in axs:
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.6)
    

    mean_cost_traditional_solver = np.mean(cost_traditional_solver, axis=1) 
    mean_accuracy_traditional_solver = np.mean(accuracy_traditional_solver, axis=1)    
    std_cost_traditional_solver  = np.std(cost_traditional_solver, axis=1, ddof=1)       
    std_accuracy_traditional_solver  = np.std(accuracy_traditional_solver, axis=1, ddof=1)

    mean_cost_traditional_solver_etd1 = np.mean(cost_traditional_solver_etd1, axis=1) 
    mean_accuracy_traditional_solver_etd1 = np.mean(accuracy_traditional_solver_etd1, axis=1)    
    std_cost_traditional_solver_etd1  = np.std(cost_traditional_solver_etd1, axis=1, ddof=1)       
    std_accuracy_traditional_solver_etd1  = np.std(accuracy_traditional_solver_etd1, axis=1, ddof=1)

    mean_cost_traditional_solver_etdrk4 = np.mean(cost_traditional_solver_etdrk4, axis=1) 
    mean_accuracy_traditional_solver_etdrk4 = np.mean(accuracy_traditional_solver_etdrk4, axis=1)    
    std_cost_traditional_solver_etdrk4  = np.std(cost_traditional_solver_etdrk4, axis=1, ddof=1)       
    std_accuracy_traditional_solver_etdrk4  = np.std(accuracy_traditional_solver_etdrk4, axis=1, ddof=1)

    mean_cost_mno_solver = np.mean(cost_mno_solver, axis=1)  
    mean_accuracy_mno_solver = np.mean(accuracy_mno_solver, axis=1)    
    std_cost_mno_solver  = np.std(cost_mno_solver, axis=1, ddof=1)       
    std_accuracy_mno_solver  = np.std(accuracy_mno_solver, axis=1, ddof=1)

    print("mean_cost_mno_solver: ", mean_cost_mno_solver)
    print("mean_accuracy_mno_solver:", mean_accuracy_mno_solver)

    # axs[0].loglog(mean_accuracy_traditional_solver, mean_cost_traditional_solver[...,0], 'o-')
    axs[0].errorbar(mean_accuracy_traditional_solver[...,0], mean_cost_traditional_solver[...,0], xerr=std_accuracy_traditional_solver[...,0], fmt='s', color='C0')
    log_err = np.log10(mean_accuracy_traditional_solver[...,0])[1:]
    log_cost = np.log10(mean_cost_traditional_solver[...,0])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[0].loglog(10**x_fit, 10**y_fit, '--', color='C0', linewidth=2,
                label=f'RK4 ($\\varepsilon^{{{slope:.2f}}}$)')
        
    # axs[0].loglog(mean_accuracy_mno_solver, mean_cost_mno_solver[...,0], 'o-')
    print(mean_accuracy_mno_solver.shape, mean_cost_mno_solver.shape, std_accuracy_mno_solver.shape)

    axs[0].errorbar(mean_accuracy_traditional_solver_etd1[...,0], mean_cost_traditional_solver_etd1[...,0], xerr=std_accuracy_traditional_solver_etd1[...,0], fmt='p', color=light_blue)
    log_err = np.log10(mean_accuracy_traditional_solver_etd1[...,0])[1:]
    log_cost = np.log10(mean_cost_traditional_solver_etd1[...,0])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[0].loglog(10**x_fit, 10**y_fit, '--', color=light_blue, linewidth=2,
                label=f'ETD1 ($\\varepsilon^{{{slope:.2f}}}$)')
                # label=f'Spectral method ETD1 ($\\varepsilon^{{{slope:.2f}}}$)')

    print("mean_accuracy_traditional_solver_etd1 :", mean_accuracy_traditional_solver_etd1)

    axs[0].errorbar(mean_accuracy_traditional_solver_etdrk4[...,0], mean_cost_traditional_solver_etdrk4[...,0], xerr=std_accuracy_traditional_solver_etdrk4[...,0], fmt='x', color=dark_blue)
    log_err = np.log10(mean_accuracy_traditional_solver_etdrk4[...,0])[1:]
    log_cost = np.log10(mean_cost_traditional_solver_etdrk4[...,0])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[0].loglog(10**x_fit, 10**y_fit, '--', color=dark_blue, linewidth=2,
                label=f'ETDRK4 ($\\varepsilon^{{{slope:.2f}}}$)')
                # label=f'Spectral method etdrk4 ($\\varepsilon^{{{slope:.2f}}}$)')



    axs[0].errorbar(mean_accuracy_mno_solver[...,0], mean_cost_mno_solver[...,0], xerr=std_accuracy_mno_solver[...,0], fmt='o', color='C1')
    log_err = np.log10(mean_accuracy_mno_solver[...,0]) 
    log_cost = np.log10(mean_cost_mno_solver[...,0]) 
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[0].loglog(10**x_fit, 10**y_fit, '--', color='C1', linewidth=2,
                label=f'NO ($\\varepsilon^{{{slope:.2f}}}$)' if nt_error==1 else 'NO')
    
    axs[0].set_xlabel(r"Rel. $L^2$ error")
    axs[0].set_ylabel("Floating-point cost")
    axs[0].legend(loc='lower left')
    axs[0].set_ylim(bottom=1e8 if nt_error == 30 else 1e7)
    # axs[1].loglog(mean_accuracy_traditional_solver, mean_cost_traditional_solver[...,1], 'o-')
    set_accuracy_ticks(axs[0])
    
    axs[1].errorbar(mean_accuracy_traditional_solver[...,0], mean_cost_traditional_solver[...,2], xerr=std_accuracy_traditional_solver[...,0], yerr=std_cost_traditional_solver[...,2], fmt='s', color='C2')
    log_err = np.log10(mean_accuracy_traditional_solver[...,0])[1:]
    log_cost = np.log10(mean_cost_traditional_solver[...,2])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color='C2', linewidth=2,
                label=f'RK4 (GPU)')
    


    axs[1].errorbar(mean_accuracy_traditional_solver[...,0], mean_cost_traditional_solver[...,1], xerr=std_accuracy_traditional_solver[...,0], yerr=std_cost_traditional_solver[...,1], fmt='s', color='C0')
    log_err = np.log10(mean_accuracy_traditional_solver[...,0])[1:]
    log_cost = np.log10(mean_cost_traditional_solver[...,1])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color='C0', linewidth=2,
                label=f'RK4 (CPU)')
    

    axs[1].errorbar(mean_accuracy_traditional_solver_etd1[...,0], mean_cost_traditional_solver_etd1[...,2], xerr=std_accuracy_traditional_solver_etd1[...,0], yerr=std_cost_traditional_solver_etd1[...,2], fmt='p', color=light_green)
    log_err = np.log10(mean_accuracy_traditional_solver_etd1[...,0])[1:]
    log_cost = np.log10(mean_cost_traditional_solver_etd1[...,2])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color=light_green, linewidth=2,
                label=f'ETD1 (GPU)')
    


    axs[1].errorbar(mean_accuracy_traditional_solver_etd1[...,0], mean_cost_traditional_solver_etd1[...,1], xerr=std_accuracy_traditional_solver_etd1[...,0], yerr=std_cost_traditional_solver_etd1[...,1], fmt='p', color=light_blue)
    log_err = np.log10(mean_accuracy_traditional_solver_etd1[...,0])[1:]
    log_cost = np.log10(mean_cost_traditional_solver_etd1[...,1])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color=light_blue, linewidth=2,
                label=f'ETD1 (CPU)')



    axs[1].errorbar(mean_accuracy_traditional_solver_etdrk4[...,0], mean_cost_traditional_solver_etdrk4[...,2], xerr=std_accuracy_traditional_solver_etdrk4[...,0], yerr=std_cost_traditional_solver_etdrk4[...,2], fmt='x', color=dark_green)
    log_err = np.log10(mean_accuracy_traditional_solver_etdrk4[...,0])[1:]
    log_cost = np.log10(mean_cost_traditional_solver_etdrk4[...,2])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color=dark_green, linewidth=2,
                label=f'ETDRK4 (GPU)')
    


    axs[1].errorbar(mean_accuracy_traditional_solver_etdrk4[...,0], mean_cost_traditional_solver_etdrk4[...,1], xerr=std_accuracy_traditional_solver_etdrk4[...,0], yerr=std_cost_traditional_solver_etdrk4[...,1], fmt='x', color=dark_blue)
    log_err = np.log10(mean_accuracy_traditional_solver_etdrk4[...,0])[1:]
    log_cost = np.log10(mean_cost_traditional_solver_etdrk4[...,1])[1:]
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color=dark_blue, linewidth=2,
                label=f'ETDRK4 (CPU)')




    axs[1].errorbar(mean_accuracy_mno_solver[...,0], mean_cost_mno_solver[...,2], xerr=std_accuracy_mno_solver[...,0], yerr=std_cost_mno_solver[...,2], fmt='o', color='C1')
    log_err = np.log10(mean_accuracy_mno_solver[...,0])
    log_cost = np.log10(mean_cost_mno_solver[...,2])
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color='C1', linewidth=2,
                label=f'NO (GPU)')

    axs[1].errorbar(mean_accuracy_mno_solver[...,0], mean_cost_mno_solver[...,1], xerr=std_accuracy_mno_solver[...,0], yerr=std_cost_mno_solver[...,1], fmt='o', color='C3')
    log_err = np.log10(mean_accuracy_mno_solver[...,0])
    log_cost = np.log10(mean_cost_mno_solver[...,1])
    slope, intercept = np.polyfit(log_err, log_cost, 1)
    x_fit = np.linspace(min(log_err), max(log_err), 50)
    y_fit = slope * x_fit + intercept
    axs[1].loglog(10**x_fit, 10**y_fit, '--', color='C3', linewidth=2,
                label=f'NO (CPU)')
    set_accuracy_ticks(axs[1])

    
    axs[1].set_xlabel(r"Rel. $L^2$ error")
    axs[1].set_ylabel("Runtime (s)")
    axs[1].legend(loc='lower left')
    

    
    fig.suptitle(f'T={nt_error}', y=0.92) 
    fig.tight_layout()
    fig.savefig(f"figs/navier_stokes_cost_accuracy_{nt_error}.png")    




if __name__ == "__main__":
    visualize_data(visualize_prediction=True)
    cost_accuracy_onestep_plot()
    cost_accuracy_plot(nt_error=30)
    nrollouts_plot()
