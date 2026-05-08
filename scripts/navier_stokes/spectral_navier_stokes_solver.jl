using GeophysicalFlows
using FourierFlows
using NPZ
using Random
using CUDA
using Printf
using LinearAlgebra
using Base.Threads

function solve(nx, ny, Lx, Ly, ν, ζ0, F_hat, dt, Tsave, nsaves = 101, dev = CPU(); verbose = false )
    # ============================================================
    # User parameters
    # ============================================================

    nν = 1
    μ  = 0.0
    nμ = 0
    stepper = "RK4"              # explicit RK4
    

    # ============================================================
    # Deterministic Kolmogorov forcing
    # Build forcing in physical space once, FFT once, reuse every step.
    # ============================================================

    function calcF!(Fh, sol, t, clock, vars, params, grid)
        @. Fh = F_hat
        return nothing
    end

    # ============================================================
    # Build forced problem
    # ============================================================
    prob = TwoDNavierStokes.Problem(
        dev;
        nx=nx, ny=ny,
        Lx=Lx, Ly=Ly,
        ν=ν, nν=nν,
        μ=μ, nμ=nμ,
        dt=dt,
        stepper=stepper,
        calcF=calcF!,
    )

    sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid


    TwoDNavierStokes.set_ζ!(prob, device_array(dev)(ζ0))
    TwoDNavierStokes.updatevars!(prob)

    # ============================================================
    # Storage: zeta[m+1, :, :] = vorticity at time m*Tsave
    # ============================================================
    times = zeros(Float64, nsaves)
    zeta_data = Array{Float64}(undef, nsaves, nx, ny)

    times[1] = 0.0
    zeta_data[1, :, :] .= Array(vars.ζ)

    save_every_steps = round(Int, Tsave / dt)
    @assert isapprox(save_every_steps * dt, Tsave; atol=1e-12) "Choose Tsave compatible with dt."

    # ============================================================
    # Time stepping and saving
    # ============================================================
    for m in 1:(nsaves - 1)
        stepforward!(prob, save_every_steps)
        TwoDNavierStokes.updatevars!(prob)

        times[m + 1] = clock.t
        zeta_data[m + 1, :, :] .= Array(vars.ζ)

        if verbose
            println("saved snapshot ", m, " at t = ", clock.t)
        end
    end

    return zeta_data
end


function taylor_green_vortex_test()
    # -----------------------------
    # parameters
    # -----------------------------
    nx       = 128
    ny       = 64
    L       = 2π
    ν       = 1e-2
    dt      = 1e-3
    T       = 1.0
    
    U0      = 1.0
    k       = 1          # mode number
    dev     = CPU()
    

    # -----------------------------
    # exact Taylor–Green vortex
    # -----------------------------
    # GeophysicalFlows examples use real-space arrays for vars.ζ and initialize
    # with set_ζ!(prob, ζ₀). We build ζ₀ on the periodic grid.
    x, y = LinRange(0, L, nx+1)[1:end-1], LinRange(0, L, ny+1)[1:end-1]
    ζ0 = [2 * k * U0 * sin(k * xx) * sin(k * yy) for xx in x, yy in y]

    # exact vorticity at time T
    decay = exp(-2 * ν * k^2 * T)
    ζ_exact = [2 * k * U0 * decay * sin(k * xx) * sin(k * yy) for xx in x, yy in y]

    F_phys = zeros(nx,ny)
    F_phys_dev = device_array(dev)(F_phys)
    F_hat = rfft(F_phys_dev)

                     
    zeta_data = solve(nx, ny, L, L, ν, ζ0, F_hat, dt, T, 2, dev; verbose = true)

    # -----------------------------
    # errors
    # -----------------------------
    err = zeta_data[end,:,:] .- ζ_exact
    l2err = norm(err) / norm(ζ_exact)
    linferr = maximum(abs.(err))


    @printf("t        = %.6f\n", T)
    @printf("L2 rel   = %.6e\n", l2err)
    @printf("Linf abs = %.6e\n", linferr)

end

function generate_data(;nx = 256, ny = 256, ndata = 10)
    Lx = 1.0
    Ly = 1.0
    
    # Standard incompressible 2D Navier–Stokes viscosity
    ν  = 1e-4
    dt = 1e-3
    Tsaves = 1.0
    nsaves = 51

    dev     = CPU()

    # Forcing parameters
    # vorticity forcing: F = 0.1 ( cos [2 pi (x + y)]  +  sin [2 pi (x + y)] )
    x, y = LinRange(0, Lx, nx+1)[1:end-1], LinRange(0, Ly, ny+1)[1:end-1]
    F_phys = 0.1*[cos(2*pi*(xx + yy)) + sin(2*pi*(xx + yy)) for xx in x, yy in y]
    F_phys_dev = device_array(dev)(F_phys)
    F_hat = rfft(F_phys_dev)
    

    zeta0_data = NPZ.npzread("../../data/navier_stokes/navier_stokes_zeta0.npy")
    @threads for i = 1:ndata
        ζ0 = zeta0_data[i,:,:]       
        zeta_data = solve(nx, ny, Lx, Ly, ν, ζ0, F_hat, dt, Tsaves, nsaves, dev)
        # ============================================================
        # Save to NumPy-compatible .npz
        # ============================================================

        if any(isnan, zeta_data)
            # Print the index (thread-safe, but output might interleave – that's fine)
            println("NaN detected in iteration i = ", i)
        end


        NPZ.npzwrite(@sprintf("../../data/navier_stokes/navier_stokes_%05d.npy", i-1), vcat(reshape(F_phys, 1, nx, ny),zeta_data))
    end

end


function cost_accuracy_traditional_solver_helper(device, n_downsample, n_trial)
    """
    Traditional solver error .
    """
    # load reference solution
    
    nt = 50  # number of iterations
    dev = device == "cpu" ? CPU() : GPU()
    ν = 1e-4

    nx = ny = 256
    dt = 1/512.0

    Tsaves = 1.0
    L = 1.0
    cost, accuracy = zeros(n_downsample, n_trial, 2), zeros(n_downsample, n_trial, nt+1)
    sol = []
    for downsample = 0:n_downsample-1 
        stride = 2^downsample
        for i = 1:n_trial
            data = NPZ.npzread(@sprintf("../../data/navier_stokes/navier_stokes_%05d.npy", i-1))
            # data : nt+2 by n by n array. 
            # F, w_0, w_1, ... , w_nt
            
            data = data[:, 1:stride:end, 1:stride:end]
            F_phys, zeta_data_ref  = data[1, :,:], data[2:end,:,:]
            F_phys_dev = device_array(dev)(F_phys)
            F_hat = rfft(F_phys_dev)

            ζ0 = zeta_data_ref[1,:,:]

            n, _ = size(F_phys)            # number of cells in each direction
            ne = n * n
            start_time = time()
            zeta_data = solve(div(nx,stride), div(ny,stride), L, L, ν, ζ0, F_hat, dt*stride, Tsaves, nt+1, dev; verbose = false)
            end_time = time()
            
            
            
            
            cost_cpu_time = end_time - start_time
            cost[downsample+1, i, :] .=  [nt*Tsaves/(dt*stride)*(100*ne*log2(ne) + 184*ne), cost_cpu_time]            
            rel_error = [norm(zeta_data[j,:,:] - zeta_data_ref[j,:,:])/norm(zeta_data_ref[j,:,:]) for j = 1:nt+1]
            accuracy[downsample+1, i, :] = rel_error
            print("relative error is : ", rel_error, " cpu_time = ", cost_cpu_time, "\n")

            if i == 1 # save data
                push!(sol, cat(reshape(F_phys, 1, n, n), zeta_data_ref, zeta_data; dims=1))
            end
        end
    end



    return  cost, accuracy, sol 

end


function cost_accuracy_traditional_solver(;n_downsample, n_trial)
    """
    Traditional solver error .
    """
    nt = 50
    # load reference solution
    # floating point cost, CPU, GPU
    cost, accuracy = zeros(n_downsample, n_trial, 3), zeros(n_downsample, n_trial, 2, nt+1)
    

    
    for device in ["cpu", "gpu"]
        cost_ds, accuracy_ds, sol_ds =  cost_accuracy_traditional_solver_helper(device, n_downsample, n_trial)
        cost[:, :, 1] = cost_ds[:,:,1]
        if device == "cpu"
            cost[:, :, 2] = cost_ds[:,:,2]
            accuracy[:, :, 1, :] = accuracy_ds 
        else
            cost[:, :, 3] = cost_ds[:,:,2]
            accuracy[:, :, 2, :] = accuracy_ds 
        end
    end

    save_data = Dict{String, Any}()
    save_data["cost"] = cost
    save_data["accuracy"] = accuracy

    NPZ.npzwrite("data/cost_accuracy_traditional_solver_data.npz", save_data)
    
    return  cost, accuracy

end



# taylor_green_vortex_test()
# generate_data(nx = 256, ny = 256, ndata = 2000)
cost_accuracy_traditional_solver(n_downsample=4, n_trial=2)
