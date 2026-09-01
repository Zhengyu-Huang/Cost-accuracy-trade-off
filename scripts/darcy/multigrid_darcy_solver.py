"""Provide Firedrake Darcy solvers, data generation, and FEM benchmarks.

Run from ``scripts/darcy`` with Firedrake/PETSc available. The active driver
requires ``../../data/darcy/darcy_data_09999.npy`` and writes the representative
solution to ``data/traditional_solver_data.npz``; the other workflows are
exposed as importable functions and are not selected by the default entry point.
Example: ``python3 multigrid_darcy_solver.py``.
"""

import os, sys
from pathlib import Path
import time
import numpy as np
from firedrake import *
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import gc


# Add the project root so shared utility modules can be imported.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utility.gaussian_random_fields import gaussian_random_field_2d

def nodal_array_to_function(arr, function_space):
    """
    Convert a structured nodal array to a Firedrake CG1 Function.

    Parameters
    ----------
    arr : np.ndarray
        Array indexed as ``arr[x_index, y_index]`` on the unit square.
    function_space : firedrake.FunctionSpace
        CG1 space whose mesh vertices coincide with the array nodes.

    Returns
    -------
    firedrake.Function
        CG1 Function with values interpolated from the nodal array.
    """
    # Infer grid dimensions from the array shape
    ny_plus1, nx_plus1 = arr.shape
    nx = nx_plus1 - 1
    ny = ny_plus1 - 1

    # The caller supplies the CG1 space on the finest hierarchy mesh.
    V = function_space

    # Create the Function
    f = Function(V, name="from_nodal_array")
    mesh = V.mesh()
    # Get vertex coordinates in the order matching the DOFs of V
    coords = mesh.coordinates.dat.data   # shape (nvertices, 2)

    # Map each vertex (x, y) to grid indices (i, j)
    i = np.round(coords[:, 0] * nx).astype(int)   # column index (x)
    j = np.round(coords[:, 1] * ny).astype(int)   # row index (y)
    i = np.clip(i, 0, nx)
    j = np.clip(j, 0, ny)

    # Look up values from the input array
    vals = arr[i, j]

    # Assign the reordered nodal values through Firedrake's vector interface.
    with f.dat.vec_wo as v:
        v.setValues(range(len(vals)), vals)

    return f


def function_to_nodal_array(u, nx, ny):
    """
    Convert a CG1 Firedrake Function on a UnitSquareMesh(nx, ny) into a 2D numpy array
    of shape (nx+1, ny+1) with values at the vertices, ordered such that arr[i, j] corresponds
    to (x=i/nx, y=j/ny).
    """
    # Get the function space and mesh
    V = u.function_space()
    mesh = V.mesh()
    
    # Vertex coordinates in the order matching the DOFs of V (CG1)
    coords = mesh.coordinates.dat.data  # shape (nvertices, 2)
    # Values at those vertices (same order)
    vals = u.dat.data_ro
    
    # Preallocate output array
    arr = np.zeros((nx + 1, ny + 1))
    
    # Map each vertex (x, y) to grid indices i, j
    i = np.round(coords[:, 0] * nx).astype(int)   # column index (x)
    j = np.round(coords[:, 1] * ny).astype(int)   # row index (y)
    # Clip for safety (should be exact)
    i = np.clip(i, 0, nx)
    j = np.clip(j, 0, ny)
    
    # Assign values – note: if multiple vertices map to same grid point (not possible here)
    arr[i, j] = vals
    
    return arr

def solve_darcy_equation(nx, ny, hierarchy_level, kappa, f, solver_parameters = {}):
    """
    Solve the Darcy equation: -div(kappa grad u) = f on a unit square,
    with homogeneous Dirichlet boundary conditions, using a geometric multigrid preconditioner.

    The mesh is constructed via a mesh hierarchy: a coarse mesh of size
    (nx/2^hierarchy_level, ny/2^hierarchy_level) is refined `hierarchy_level` times
    to produce the final mesh of size (nx, ny).

    Parameters
    ----------
    nx, ny : int
        Number of cells in x and y directions on the finest mesh.
    hierarchy_level : int
        Number of refinement levels. The coarse mesh size is nx // 2**hierarchy_level,
        ny // 2**hierarchy_level.
    kappa : numpy.ndarray 
        Permeability coefficient. A numpy array of shape (nx+1, ny+1) is provided,
        it is converted to a CG1 Function on the finest mesh using nodal_array_to_function.
    f : numpy.ndarray 
        Source term. Handled similarly to kappa.
    solver_parameters : dict, optional
        PETSc solver parameters (e.g., {"ksp_type": "cg", "pc_type": "ilu"}).
        
    Returns
    -------
    uh : firedrake.Function
        The computed solution (CG1 function on the finest mesh).
    V : firedrake.FunctionSpace
        The function space on the finest mesh.
    """
    
    # ------------------------------------------------------------------
    # 1. Build mesh hierarchy
    # ------------------------------------------------------------------
    # Coarse mesh size must be integer; ensure nx and ny are divisible by 2**hierarchy_level            
        
    mesh0 = UnitSquareMesh(nx // 2**hierarchy_level, ny // 2**hierarchy_level, quadrilateral=True)
    hierarchy = MeshHierarchy(mesh0, hierarchy_level)
    mesh = hierarchy[-1]

    # ------------------------------------------------------------------
    # 2. Function space and coordinates
    # ------------------------------------------------------------------
    V = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)

    # ------------------------------------------------------------------
    # 3. Convert kappa and f to Functions on V (if they are not already)
    # ------------------------------------------------------------------
    # nodal_array_to_function is assumed to exist (as defined previously)
    kappa_func = nodal_array_to_function(kappa, function_space=V)
    f_func = nodal_array_to_function(f, function_space=V)
    
    
    # trial/test
    u = TrialFunction(V)
    v = TestFunction(V)

    # bilinear and linear forms
    a = inner(kappa_func * grad(u), grad(v)) * dx
    L = f_func * v * dx

    # Homogeneous Dirichlet boundary condition.
    bc = DirichletBC(V, 0.0, "on_boundary")

    # solution
    uh = Function(V, name="uh")

    # ------------------------------------------------------------------
    # 4. Solve with the requested multigrid preconditioner
    # ------------------------------------------------------------------
    # Exclude one warm-up solve, then reset the iterate before every timed solve.
    solve(a == L, uh, bcs=bc, solver_parameters=solver_parameters)
    n_repeat = 10
    start_time = time.perf_counter()
    for i in range(n_repeat):
        uh.assign(0) 
        solve(a == L, uh, bcs=bc, solver_parameters=solver_parameters)
    end_time = time.perf_counter()
    
    
    solve_time = (end_time - start_time)/n_repeat
    print(f"Solve time for {nx}x{ny} mesh: {solve_time:.4f} seconds")
    
    return uh, V, solve_time


def test_darcy_equation():
    """Check the multigrid solve against a manufactured smooth solution."""
    solver_parameters_cg = {
        "ksp_type": "cg",               # Conjugate Gradient (optimal for SPD)
        "pc_type": "mg",                # geometric multigrid
        "pc_mg_cycle_type": "v",        # V-cycle (cheapest, usually sufficient)
        "mg_levels_ksp_type": "chebyshev",  # Chebyshev smoothing (better than Richardson)
        "mg_levels_pc_type": "jacobi",  # Jacobi preconditioner for Chebyshev
        "mg_coarse_ksp_type": "preonly",
        "mg_coarse_pc_type": "lu",      # Direct solve on coarse grid
        "ksp_rtol": 1e-8,               # relative tolerance
        "ksp_atol": 1e-12,              # absolute tolerance
        "ksp_max_it": 200,              # safeguard
        "ksp_monitor": None,            # optional: print residual history
        "ksp_converged_reason": None,   # optional: print convergence reason
    }
    
    solver_parameters_mg = {
        "ksp_type": "richardson",
        #"ksp_max_it": 10,
        #"ksp_rtol": 1.0e-5,
        "pc_type": "mg",
        "pc_mg_type": "multiplicative",
        "pc_mg_cycle_type": "v",
        "mg_levels_ksp_type": "richardson",
        "mg_levels_pc_type": "jacobi",
        "mg_coarse_ksp_type": "preonly",
        "mg_coarse_pc_type": "lu",
        "ksp_monitor": None,
        "ksp_converged_reason": None,
    }
    
    
    for (nx, ny, hierarchy_level) in [(256, 256, 4), (512, 512, 5)]:
    
        x, y = np.meshgrid(np.linspace(0,1,nx+1), np.linspace(0,1,ny+1), indexing='ij')
        kappa_data = 1 + 2*x + y
        u_exact_data = np.sin(np.pi * x) * np.sin(2 * np.pi * y)
        f_data = (5 * np.pi**2 * (1 + 2*x + y) * np.sin(np.pi*x) * np.sin(2*np.pi*y) 
        - 2*np.pi * np.cos(np.pi*x) * np.sin(2*np.pi*y)
        - 2*np.pi * np.sin(np.pi*x) * np.cos(2*np.pi*y))

        uh, V, _ = solve_darcy_equation(nx, ny, hierarchy_level, kappa_data, f_data, solver_parameters = solver_parameters_mg)
        
        # function space
        mesh = V.mesh()
        # coordinates
        x_expr, y_expr = SpatialCoordinate(mesh)
        # exact solution
        u_exact_expr = sin(pi * x_expr) * sin(2 * pi * y_expr)
        
        # interpolate exact solution for error computation
        u_exact = Function(V, name="u_exact").interpolate(u_exact_expr)
        err = Function(V, name="error")
        err.assign(uh - u_exact)

        # errors
        print("Resolution nx, ny = ", nx, ny)
        L2_err = norm(err, norm_type="L2")
        H1_err = norm(err, norm_type="H1")
        print(f"L2 error  = {L2_err:.12e}")
        print(f"H1 error  = {H1_err:.12e}")
        uh_data = function_to_nodal_array(uh, nx, ny)
        Rel_L2_err = np.linalg.norm(uh_data - u_exact_data)/np.linalg.norm(u_exact_data)
        print(f"Rel_L2_err  = {Rel_L2_err:.12e}")
        
        
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        im = axs[0].pcolormesh(x, y, kappa_data)
        axs[0].set_title("kappa");axs[0].set_aspect('equal')
        fig.colorbar(im, ax=axs[0])
        im = axs[1].pcolormesh(x, y, f_data)
        axs[1].set_title("f");axs[1].set_aspect('equal')
        fig.colorbar(im, ax=axs[1])
        im = axs[2].pcolormesh(x, y, uh_data)
        axs[2].set_title("u (predicted)");axs[2].set_aspect('equal')
        fig.colorbar(im, ax=axs[2])
        im = axs[3].pcolormesh(x, y, u_exact_data)
        axs[3].set_title("u (reference)");axs[3].set_aspect('equal')
        fig.colorbar(im, ax=axs[3])
        fig.savefig(f"figs/Darcy_flow_{nx}_{ny}.png")
        


        


    
    
def generate_data():
    """
    Generate synthetic Darcy flow data: random permeability fields (kappa)
    and corresponding pressure solutions (u) for a 2D Darcy equation.
    Saves each (kappa, u) pair as a separate .npy file.
    """
        
    Path('../../data/darcy').mkdir(parents=True, exist_ok=True)
    
    solver_parameters_mg = {
        "ksp_type": "richardson",
        "pc_type": "mg",
        "pc_mg_type": "multiplicative",
        "pc_mg_cycle_type": "v",
        "mg_levels_ksp_type": "richardson",
        "mg_levels_pc_type": "jacobi",
        "mg_coarse_ksp_type": "preonly",
        "mg_coarse_pc_type": "lu",
        "ksp_monitor": None,
        "ksp_converged_reason": None,
    }
        
    ndata = 10000
    nx = ny = 512
    ngrid = nx + 1
    L = 1.0
    
    f_data = np.ones((ngrid, ngrid))
    for i in range(ndata):
        kappa_data = gaussian_random_field_2d(1, [ngrid, ngrid], [L, L], sigma=1.0, tau = 3.0, alpha = 2.0, bc_name = 'neumann', seed = i)
        kappa_data = kappa_data[0,...]
        # Threshold the Gaussian field into the two permeability phases 1 and 10.
        positive_indices = kappa_data >= 0
        kappa_data[positive_indices] = 10
        kappa_data[~positive_indices] = 1
        
        uh, V, _ = solve_darcy_equation(nx, ny, hierarchy_level=6, kappa = kappa_data, f = f_data, solver_parameters = solver_parameters_mg)
        u_data = function_to_nodal_array(uh, nx, ny)
        
        np.save(f"../../data/darcy/darcy_data_{i:05d}.npy", np.stack([kappa_data, u_data], axis=-1))




def visualize_data():
    """Plot permeability, forcing, and solution from one generated sample."""
        
    i = 1
    data = np.load(f"../../data/darcy/darcy_data_{i:05d}.npy")
    kappa_data, u_data = data[:,:,0], data[:,:,1]
    
    ngrid, _ = u_data.shape
    f_data = np.ones((ngrid, ngrid))
    x, y = np.meshgrid(np.linspace(0,1,ngrid), np.linspace(0,1,ngrid), indexing='ij')
            
    
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    im = axs[0].pcolormesh(x, y, kappa_data)
    axs[0].set_title("kappa");axs[0].set_aspect('equal')
    fig.colorbar(im, ax=axs[0])
    im = axs[1].pcolormesh(x, y, f_data)
    axs[1].set_title("f");axs[1].set_aspect('equal')
    fig.colorbar(im, ax=axs[1])
    im = axs[2].pcolormesh(x, y, u_data)
    axs[2].set_title("u (predicted)");axs[2].set_aspect('equal')
    fig.colorbar(im, ax=axs[2])
    fig.savefig(f"figs/Darcy_flow_{i}.png")
    
    
    
def cost_accuracy_traditional_solver():
    """Benchmark FEM work, runtime, and relative error across six grids.

    Cost has shape ``(resolution, trial, 2)`` with channels
    ``[analytical FLOP estimate, CPU seconds]``. Accuracy stores relative L2
    error against the downsampled 512-by-512-cell reference solution.
    """
    
    m_iteration = 15  # modeled number of multigrid V-cycles
    nu_iteration = 2  # modeled smoothing steps per V-cycle
    
    solver_parameters_mg = {
        "ksp_type": "richardson",
        "ksp_max_it": m_iteration,
        "ksp_rtol": 1.0e-6,
        "pc_type": "mg",
        "pc_mg_type": "multiplicative",
        "pc_mg_cycle_type": "v",
        "mg_levels_ksp_type": "richardson",
        "mg_levels_pc_type": "jacobi",
        "mg_coarse_ksp_type": "preonly",
        "mg_coarse_pc_type": "lu",
        # "ksp_monitor": None,
        # "ksp_converged_reason": None,
    }
    
    nx = ny = 512
    ngrid = nx + 1
    L = 1.0
    n_downsample, n_trial = 6, 10
    cost, accuracy = np.zeros((n_downsample, n_trial, 2)), np.zeros((n_downsample, n_trial))
    sol = []
    # Increasing downsample coarsens the nodal grid by powers of two.
    for downsample in range(6):
        
        for i in range(n_trial):
            # Use the final n_trial samples, in reverse file order.
            data = np.load(f"../../data/darcy/darcy_data_{(9999-i):05d}.npy")
            print(data.shape)
            # The final axis stores permeability and the reference solution.
            stride = 2**downsample
            data = data[0::stride, 0::stride, :]
            kappa_data, u_ref  = data[:,:,0], data[:,:,1]
            n_plus1, _ = kappa_data.shape  # number of point in each direction
            n = n_plus1 - 1                # number of element in each direction
            f_data = np.ones((n, n))
            
            uh, V, cost_cpu_time = solve_darcy_equation(n, n, hierarchy_level=6-downsample, kappa = kappa_data, f = f_data, solver_parameters = solver_parameters_mg)
            u_data = function_to_nodal_array(uh, n, n)
            
            
            
            rel_error = np.linalg.norm(u_data - u_ref)/np.linalg.norm(u_ref)
            # Work model: setup plus m V-cycles, including nu smoothing steps.
            cost[downsample, i, :] =  (271 + m_iteration*(88/3 * nu_iteration + 116/3))*n*n, cost_cpu_time
            accuracy[downsample, i] = rel_error
            print("relative error is : ", rel_error, " cpu_time = ", cost_cpu_time)

            if i == 0: # retain one representative field triplet per resolution
                sol.append(np.stack([kappa_data, u_ref, u_data], axis=2))
    
    np.savez_compressed('data/cost_accuracy_traditional_solver_data.npz', cost=cost, accuracy=accuracy, sol=np.array(sol, dtype=object))

    return  cost, accuracy, sol 





def traditional_solver(test_index, downsample):
    """Solve one downsampled field with the fixed FEM configuration and save it."""
    
    m_iteration = 15  # number of multigrid v cycle iterations
    nu_iteration = 2  # number of smoothing per multigrid v cycle iteration
    
    solver_parameters_mg = {
        "ksp_type": "richardson",
        "ksp_max_it": m_iteration,
        "ksp_rtol": 1.0e-6,
        "pc_type": "mg",
        "pc_mg_type": "multiplicative",
        "pc_mg_cycle_type": "v",
        "mg_levels_ksp_type": "richardson",
        "mg_levels_pc_type": "jacobi",
        "mg_coarse_ksp_type": "preonly",
        "mg_coarse_pc_type": "lu",
        # "ksp_monitor": None,
        # "ksp_converged_reason": None,
    }
    
    nx = ny = 512
    ngrid = nx + 1
    L = 1.0
    data = np.load(f"../../data/darcy/darcy_data_{test_index:05d}.npy")
            
    # The final axis stores permeability and the reference solution.
    stride = 2**downsample
    data = data[0::stride, 0::stride, :]
    kappa_data, u_ref  = data[:,:,0], data[:,:,1]
    n_plus1, _ = kappa_data.shape  # number of point in each direction
    n = n_plus1 - 1                # number of element in each direction
    f_data = np.ones((n, n))
    
    uh, V, cost_cpu_time = solve_darcy_equation(n, n, hierarchy_level=6-downsample, kappa = kappa_data, f = f_data, solver_parameters = solver_parameters_mg)
    u_data = function_to_nodal_array(uh, n, n)
            
    
    np.savez_compressed('data/traditional_solver_data.npz', sol=u_data)

    return  



if __name__ == "__main__":
    
    
    #############################################
    # Test Firedrake solver with exact solution
    #############################################
    
    test_darcy_equation()
        
        
    #############################################
    # Generate 10000 data
    #############################################
    
    # generate_data()
    
    
    
    #############################################
    # Visualize the generated data (default data id 1)
    #############################################
    
    # visualize_data()


    #############################################
    # Evaluate cost and accuracy for Firedrake solver with different mesh sizes
    #############################################
    
    # cost_accuracy_traditional_solver()
    
    
    #############################################
    # Load generated data, compute the solution at a coarser mesh 
    #############################################
    
    # traditional_solver(test_index=9999, downsample=3)
