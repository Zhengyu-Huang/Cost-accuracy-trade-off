from firedrake import *

# Example: permeability given on an nx x ny uniform grid, one value per cell
nx,ny = 64, 64
perm = np.ones((nx,ny))                     # shape (nx, ny)
nx, ny = perm.shape

# Choose a coarse mesh and number of refinements so the finest mesh matches (nx, ny)
# Example: if nx = ny = 64, take 8 x 8 coarse mesh and 3 refinements
mesh0 = UnitSquareMesh(nx // 2**3, ny // 2**3, quadrilateral=True)
hierarchy = MeshHierarchy(mesh0, 3)
mesh = hierarchy[-1]

# Solution space
V = FunctionSpace(mesh, "CG", 1)

# Cellwise constant permeability
K = FunctionSpace(mesh, "DQ", 0)
kappa = Function(K, name="permeability")

# Safer than assuming raw cell ordering: evaluate the grid data at the DQ0 dof locations
W = VectorFunctionSpace(mesh, K.ufl_element())
X = assemble(interpolate(mesh.coordinates, W)).dat.data_ro

# Map dof locations to grid-cell indices
# Depending on your array convention, you may need perm[iy, ix] instead of perm[ix, iy]
ix = np.clip((X[:, 0] * nx).astype(int), 0, nx - 1)
iy = np.clip((X[:, 1] * ny).astype(int), 0, ny - 1)
kappa.dat.data[:] = perm[ix, iy]

u = TrialFunction(V)
v = TestFunction(V)
f = Constant(100.0)

a = inner(kappa * grad(u), grad(v)) * dx
L = f * v * dx
bc = DirichletBC(V, 0.0, "on_boundary")

uh = Function(V, name="pressure")

solve(a == L, uh, bcs=bc,
      solver_parameters={
          "ksp_type": "cg",
          "pc_type": "mg",
          "mg_levels_ksp_type": "richardson",
          "mg_levels_pc_type": "jacobi",
          "mg_coarse_ksp_type": "preonly",
          "mg_coarse_pc_type": "lu",
      })


solve(a == L, uh, bcs=bc,
      solver_parameters={
          "ksp_type": "richardson",
          "pc_type": "mg",
          "mg_levels_ksp_type": "richardson",
          "mg_levels_pc_type": "jacobi",
          "mg_coarse_ksp_type": "preonly",
          "mg_coarse_pc_type": "lu",
      })


solve(a == L, uh, bcs=bc,
      solver_parameters={
          "ksp_type": "richardson",
          "ksp_max_it": 10,
          "ksp_rtol": 1.0e-4,
          "ksp_atol": 1.0e-12,

          "pc_type": "mg",
          "pc_mg_type": "multiplicative",
          "pc_mg_cycle_type": "v",

          "mg_levels_ksp_type": "richardson",
          "mg_levels_ksp_max_it": 1,
          "mg_levels_pc_type": "jacobi",

          "mg_coarse_ksp_type": "preonly",
          "mg_coarse_pc_type": "lu",

          "ksp_monitor": None,
          "ksp_converged_reason": None,
      })