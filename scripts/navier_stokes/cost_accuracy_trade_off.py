import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
plt.style.use('seaborn-v0_8-whitegrid')   # 现代网格样式
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
    
def visualize_data():
        
    i = 1999
    data = data = np.load(f"../../data/navier_stokes/navier_stokes_{i:05d}.npy")
    vorticity = data[1:, ...]
    force     = data[0, ...]
    
    ngrid, _ = force.shape
    x, y = np.meshgrid(np.linspace(0,1,ngrid, endpoint=False), np.linspace(0,1,ngrid, endpoint=False), indexing='ij')
    ts = [0,1,10,20]        
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    for i in range(4):
        im = axs[i].pcolormesh(x, y, vorticity[ts[i],...], cmap='viridis', shading='gouraud')
        axs[i].set_title(rf"$\omega (t={ts[i]})$");
        axs[i].set_xticks([]);
        axs[i].set_yticks([]);
        axs[i].set_aspect('equal')
        cbar = fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04, shrink=1.0)
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
    
    
    
def cost_accuracy_plot():
    print("darcy flow cost accuracy plot")
    cost_accuracy_traditional_solver_data = np.load('data/cost_accuracy_traditional_solver_data.npz', allow_pickle=True)   # 注意 allow_pickle=True
    cost_traditional_solver = cost_accuracy_traditional_solver_data['cost']
    accuracy_traditional_solver = cost_accuracy_traditional_solver_data['accuracy']
    
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    mean_cost_traditional_solver = np.mean(cost_traditional_solver, axis=1)     
    mean_accuracy_traditional_solver = np.mean(accuracy_traditional_solver, axis=1)    
    std_cost_traditional_solver  = np.std(cost_traditional_solver, axis=1, ddof=1)       
    std_accuracy_traditional_solver  = np.std(accuracy_traditional_solver, axis=1, ddof=1)

    # axs[0].loglog(mean_accuracy_traditional_solver, mean_cost_traditional_solver[...,0], 'o-')
    axs[0].errorbar(mean_accuracy_traditional_solver, mean_cost_traditional_solver[...,0], xerr=std_accuracy_traditional_solver, fmt='o')
    axs[0].set_xlabel("Rel. error")
    axs[0].set_ylabel("Floating-point cost")
    
    # axs[1].loglog(mean_accuracy_traditional_solver, mean_cost_traditional_solver[...,1], 'o-')
    axs[1].errorbar(mean_accuracy_traditional_solver, mean_cost_traditional_solver[...,1], xerr=std_accuracy_traditional_solver, yerr=std_cost_traditional_solver[...,1], fmt='o')
    axs[1].set_xlabel("Rel. error")
    axs[1].set_ylabel("CPU cost (s)")

    fig.tight_layout()
    fig.savefig("figs/darcy_flow_cost_accuracy.png")    
        
if __name__ == "__main__":
    # visualize_data()
    cost_accuracy_plot()