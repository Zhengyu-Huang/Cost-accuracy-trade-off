using GeophysicalFlows
using FourierFlows
using NPZ
using Random
using CUDA
using Printf
using LinearAlgebra
using Base.Threads

import FourierFlows: TimeStepper, stepforward!


"""Local first-order exponential Euler method (not provided by FourierFlows)."""
struct ETD1 end


struct ETD1TimeStepper{A, C} <: AbstractTimeStepper{A}
    N::A
    expLdt::C
    hphi1::C
end


etd1_hphi1(z, h) = iszero(z) ? h : h * expm1(z) / z


function ETD1TimeStepper(equation, dt, dev::Device = CPU())
    h = convert(eltype(equation.L), dt)
    z = h .* equation.L

    return ETD1TimeStepper(
        zeros(dev, equation.T, equation.dims),
        exp.(z),
        etd1_hphi1.(z, h),
    )
end


TimeStepper(::ETD1, equation, dt, dev::Device = CPU(); kwargs...) =
    ETD1TimeStepper(equation, dt, dev)


function stepforward!(sol, clock, ts::ETD1TimeStepper,
                      equation, vars, params, grid)
    equation.calcN!(ts.N, sol, clock.t, clock, vars, params, grid)
    @. sol = ts.expLdt * sol + ts.hphi1 * ts.N
    clock.t += clock.dt
    clock.step += 1
    return nothing
end


function time_step_flops(ne, stepper)
    if stepper == "ETD1"
        return 25 * ne * log2(ne) + 39 * ne
    elseif stepper == "ETDRK4"
        return 100 * ne * log2(ne) + 171 * ne
    else
        return 100 * ne * log2(ne) + 184 * ne
    end
end


benchmark_dt_factor(stepper) = stepper == "ETD1" ? 0.25 : 1.0


"""
    update_vorticity!(prob)

Transform only the requested vorticity field to physical space. In contrast,
`TwoDNavierStokes.updatevars!` also computes both velocity components. The inverse
FFT is part of online solution generation and is included in benchmark timings.
"""
function update_vorticity!(prob)
    dealias!(prob.sol, prob.grid)
    copyto!(prob.vars.ζh, prob.sol)
    ldiv!(prob.vars.ζ, prob.grid.rfftplan, prob.vars.ζh)
    return nothing
end


function integrate_trajectory!(prob, Tsave, nsaves; verbose = false)
    clock, vars = prob.clock, prob.vars
    dt = clock.dt
    save_every_steps = round(Int, Tsave / dt)
    nx, ny = size(vars.ζ)
    zeta_data = similar(vars.ζ, nsaves, nx, ny)
    zeta_data[1, :, :] .= vars.ζ

    for m in 1:(nsaves - 1)
        stepforward!(prob, save_every_steps)
        update_vorticity!(prob)
        zeta_data[m + 1, :, :] .= vars.ζ
        verbose && println("saved snapshot ", m, " at t = ", clock.t)
    end

    return zeta_data
end


function solve(nx, ny, Lx, Ly, ν, ζ0, F_hat, dt, Tsave,
               nsaves = 101, dev = CPU(); stepper = "RK4", verbose = false)
    nν = 1
    μ  = 0.0
    nμ = 0
    native_stepper = stepper == "ETD1" ? ETD1() : stepper

    function calcF!(Fh, sol, t, clock, vars, params, grid)
        @. Fh = F_hat
        return nothing
    end

    prob = TwoDNavierStokes.Problem(
        dev;
        nx=nx, ny=ny,
        Lx=Lx, Ly=Ly,
        ν=ν, nν=nν,
        μ=μ, nμ=nμ,
        dt=dt,
        stepper=native_stepper,
        calcF=calcF!,
    )

    TwoDNavierStokes.set_ζ!(prob, device_array(dev)(ζ0))

    dev isa GPU && CUDA.synchronize()
    start_ns = time_ns()
    zeta_data = integrate_trajectory!(prob, Tsave, nsaves; verbose)
    dev isa GPU && CUDA.synchronize()
    runtime = (time_ns() - start_ns) * 1e-9

    return Array(zeta_data), runtime
end


function taylor_green_vortex_test(; stepper = "RK4")
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

                     
    zeta_data, _ = solve(nx, ny, L, L, ν, ζ0, F_hat, dt, T, 2, dev; stepper, verbose = true)

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
        zeta_data, _ = solve(nx, ny, Lx, Ly, ν, ζ0, F_hat, dt, Tsaves, nsaves, dev)
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


function cost_accuracy_traditional_solver_helper(device, n_downsample, n_trial; stepper = "RK4")
    """
    Traditional solver error .
    """
    # load reference solution
    
    nt = 50  # number of iterations
    dev = device == "cpu" ? CPU() : GPU()
    ν = 1e-4

    nx = ny = 256
    dt = (1/512.0) * benchmark_dt_factor(stepper)

    Tsaves = 1.0
    L = 1.0
    data_dir = normpath(joinpath(@__DIR__, "..", "..", "data", "navier_stokes"))
    cost, accuracy = zeros(n_downsample, n_trial, 2), zeros(n_downsample, n_trial, nt+1)
    for downsample = 0:n_downsample-1 
        stride = 2^downsample
        steps_per_save = round(Int, Tsaves / (dt * stride))
        for i in 1:n_trial
            filename = @sprintf("navier_stokes_%05d.npy", 2000 - i)
            data = NPZ.npzread(joinpath(data_dir, filename))
            # data : nt+2 by n by n array. 
            # F, w_0, w_1, ... , w_nt
            
            data = data[:, 1:stride:end, 1:stride:end]
            F_phys, zeta_data_ref  = data[1, :,:], data[2:end,:,:]
            F_phys_dev = device_array(dev)(F_phys)
            F_hat = rfft(F_phys_dev)

            ζ0 = zeta_data_ref[1,:,:]

            n, _ = size(F_phys)            # number of cells in each direction
            ne = n * n
            # warmup
            i == 1 && solve(div(nx,stride), div(ny,stride), L, L, ν, ζ0, F_hat,
                            dt*stride, dt*stride, 2, dev; stepper)
            zeta_data, cost_cpu_time = solve(div(nx,stride), div(ny,stride), L, L, ν, ζ0, F_hat, dt*stride, Tsaves, nt+1, dev; stepper, verbose = false)
            
            
            cost[downsample+1, i, :] .= [nt * steps_per_save * time_step_flops(ne, stepper), cost_cpu_time]
            rel_error = [norm(zeta_data[j,:,:] - zeta_data_ref[j,:,:])/norm(zeta_data_ref[j,:,:]) for j = 1:nt+1]
            accuracy[downsample+1, i, :] = rel_error
            print("relative error is : ", rel_error, " cpu_time = ", cost_cpu_time, "\n")

 
        end
    end



    return  cost, accuracy 

end




function traditional_solver(;test_index, downsample, stepper = "RK4")
    """
    Traditional solver .
    """
    # load reference solution
    
    nt = 50  # number of iterations
    dev = CPU() 
    ν = 1e-4

    nx = ny = 256
    dt = (1/512.0) * benchmark_dt_factor(stepper)

    Tsaves = 1.0
    L = 1.0
    stride = 2^downsample
       
    data = NPZ.npzread(@sprintf("../../data/navier_stokes/navier_stokes_%05d.npy", test_index))
    # data : nt+2 by n by n array. 
    # F, w_0, w_1, ... , w_nt
            
    data = data[:, 1:stride:end, 1:stride:end]
    F_phys, zeta_data_ref  = data[1, :,:], data[2:end,:,:]
    F_phys_dev = device_array(dev)(F_phys)
    F_hat = rfft(F_phys_dev)

    ζ0 = zeta_data_ref[1,:,:]

    n, _ = size(F_phys)            # number of cells in each direction
    ne = n * n
    zeta_data, _ = solve(div(nx,stride), div(ny,stride), L, L, ν, ζ0, F_hat, dt*stride, Tsaves, nt+1, dev; stepper, verbose = false)

    NPZ.npzwrite("data/traditional_solver_data.npz", zeta_data)    
            
    

    return  

end



function cost_accuracy_traditional_solver(;n_downsample, n_trial, stepper = "RK4")
    """
    Traditional solver error .
    """
    nt = 50
    # load reference solution
    # floating point cost, CPU, GPU
    cost, accuracy = zeros(n_downsample, n_trial, 3), zeros(n_downsample, n_trial, 2, nt+1)
    

    
    for device in ["gpu","cpu"]
        cost_ds, accuracy_ds = cost_accuracy_traditional_solver_helper(device, n_downsample, n_trial; stepper)
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

    output = stepper == "RK4" ?
        "data/cost_accuracy_traditional_solver_data.npz" :
        "data/cost_accuracy_traditional_solver_data_$(stepper).npz"
    NPZ.npzwrite(output, save_data)
    
    return  cost, accuracy

end




function cost_accuracy_traditional_solver_onestep(;n_downsample, n_trial, stepper = "RK4")
    """
    Traditional solver error .
    """
    nt = 50
    # load reference solution
    # floating point cost, CPU, GPU
    cost, accuracy = zeros(n_downsample, n_trial, 3), zeros(n_downsample, n_trial, 2, nt)
    

    
    for device in ["gpu","cpu"]
        cost_ds, accuracy_ds = cost_accuracy_traditional_solver_onestep_helper(device, n_downsample, n_trial; stepper)
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

    output = stepper == "RK4" ?
        "data/cost_accuracy_traditional_solver_onestep_data.npz" :
        "data/cost_accuracy_traditional_solver_onestep_data_$(stepper).npz"
    NPZ.npzwrite(output, save_data)
    
    return  cost, accuracy

end

function cost_accuracy_traditional_solver_onestep_helper(device, n_downsample, n_trial; stepper = "RK4")
    """
    Traditional solver error .
    """
    # load reference solution
    
    nt = 50  # number of iterations
    dev = device == "cpu" ? CPU() : GPU()
    ν = 1e-4

    nx = ny = 256
    dt = (1/512.0) * benchmark_dt_factor(stepper)

    Tsaves = 1.0
    L = 1.0
    data_dir = normpath(joinpath(@__DIR__, "..", "..", "data", "navier_stokes"))
    cost, accuracy = zeros(n_downsample, n_trial, 2), zeros(n_downsample, n_trial, nt)
    for downsample = 0:n_downsample-1 
        stride = 2^downsample
        steps_per_save = round(Int, Tsaves / (dt * stride))
        for i in 1:n_trial
            filename = @sprintf("navier_stokes_%05d.npy", 2000 - i)
            data = NPZ.npzread(joinpath(data_dir, filename))
            # data : nt+2 by n by n array. 
            # F, w_0, w_1, ... , w_nt
            
            data = data[:, 1:stride:end, 1:stride:end]
            F_phys, zeta_data_ref  = data[1, :,:], data[2:end,:,:]
            F_phys_dev = device_array(dev)(F_phys)
            F_hat = rfft(F_phys_dev)

            

            n, _ = size(F_phys)            # number of cells in each direction
            ne = n * n
            # warmup
            i == 1 && solve(div(nx,stride), div(ny,stride), L, L, ν,
                            zeta_data_ref[1,:,:], F_hat, dt*stride, dt*stride,
                            2, dev; stepper)
            
            for j in 1:nt
                ζ0 = zeta_data_ref[j,:,:]

                zeta_data, cost_cpu_time = solve(div(nx,stride), div(ny,stride), L, L, ν, ζ0, F_hat, dt*stride, Tsaves, 2, dev; stepper, verbose = false)
            
                # one step time 
                cost[downsample+1, i, :] += [steps_per_save * time_step_flops(ne, stepper), cost_cpu_time]
                accuracy[downsample+1, i, j] = norm(zeta_data[2,:,:] - zeta_data_ref[j+1,:,:])/norm(zeta_data_ref[j+1,:,:]) 
            end

            cost[downsample+1, i, :] /= nt
 
        end
    end



    return  cost, accuracy 

end


# taylor_green_vortex_test()
# generate_data(nx = 256, ny = 256, ndata = 2000)
# traditional_solver(test_index=1999, downsample=2)
# Set stepper to "RK4", "ETD1", or "ETDRK4".
stepper="RK4"
cost_accuracy_traditional_solver(n_downsample=4, n_trial=10, stepper=stepper)
cost_accuracy_traditional_solver_onestep(n_downsample=4, n_trial=10, stepper=stepper)
