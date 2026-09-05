using GeophysicalFlows
using FourierFlows
using FFTW
using NPZ
using CUDA
using Printf
using LinearAlgebra
using Statistics
using Base.Threads

import FourierFlows: TimeStepper, stepforward!

const SCRIPT_DIR = @__DIR__
const DEFAULT_DATA_DIR = normpath(joinpath(SCRIPT_DIR, "..", "..", "data", "navier_stokes"))
const DEFAULT_OUTPUT_DIR = joinpath(SCRIPT_DIR, "data")
const SUPPORTED_STEPPERS = ("RK4", "ETDRK4", "ETD1")

"""Marker used to select the locally implemented exponential Euler method."""
struct ETD1 end

"""
    ETD1TimeStepper

Precomputed data for the first-order exponential Euler (ETD1) update

    uⁿ⁺¹ = exp(hL)uⁿ + h φ₁(hL)N(uⁿ,tⁿ),

where `φ₁(z) = expm1(z)/z` and `φ₁(0) = 1`. The diagonal linear operator is
treated exactly, and the nonlinear term is evaluated once per step.
"""
struct ETD1TimeStepper{A, C} <: AbstractTimeStepper{A}
    N::A
    expLdt::C
    hphi1::C
end

function ETD1TimeStepper(equation, dt, dev::Device = CPU())
    L_host = Array(equation.L)
    h = convert(eltype(L_host), dt)
    z = h .* L_host
    expLdt_host = exp.(z)
    hphi1_host = similar(z)

    @inbounds for i in eachindex(z)
        hphi1_host[i] = iszero(z[i]) ? h : h * expm1(z[i]) / z[i]
    end

    expLdt = device_array(dev)(expLdt_host)
    hphi1 = device_array(dev)(hphi1_host)
    N = zeros(dev, equation.T, equation.dims)
    return ETD1TimeStepper(N, expLdt, hphi1)
end

# Marker dispatch avoids FourierFlows' string-based, module-local `eval` factory.
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

function device_from_name(device::AbstractString)
    return lowercase(device) == "cpu" ? CPU() : GPU()
end

synchronize_device(::CPU) = nothing
synchronize_device(::GPU) = CUDA.synchronize()

"""
    build_problem(nx, ny, Lx, Ly, ν, F_hat, dt, dev; stepper="RK4", T=Float64)

Construct a forced two-dimensional Navier--Stokes problem. Problem construction,
FFT planning, and exponential-coefficient construction are
query-independent setup costs and are performed outside the timed region in the
benchmark below.
"""
function build_problem(nx, ny, Lx, Ly, ν, F_hat, dt, dev;
                       stepper::AbstractString = "RK4", T = Float64)
    native_stepper = stepper == "ETD1" ? ETD1() : stepper

    function calcF!(Fh, sol, t, clock, vars, params, grid)
        copyto!(Fh, F_hat)
        return nothing
    end

    return TwoDNavierStokes.Problem(
        dev;
        nx = nx,
        ny = ny,
        Lx = Lx,
        Ly = Ly,
        ν = ν,
        nν = 1,
        μ = 0,
        nμ = 0,
        dt = dt,
        stepper = native_stepper,
        calcF = calcF!,
        T = T,
    )
end

"""
Reset a reusable problem to a new physical-space vorticity field using
`TwoDNavierStokes.set_ζ!`, which transforms and dealiases the supplied field
and updates the diagnostic variables.
"""
function reset_problem!(prob, ζ0, dev)
    prob.clock.t = zero(prob.clock.t)
    prob.clock.step = 0
    ζ0_typed = convert.(eltype(prob.vars.ζ), ζ0)
    ζ0_device = device_array(dev)(ζ0_typed)
    TwoDNavierStokes.set_ζ!(prob, ζ0_device)
    return nothing
end

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

"""
    integrate_trajectory!(prob, Tsave, nsaves; verbose=false)

Advance an initialized problem and return vorticity at times
`0, Tsave, ..., (nsaves-1)Tsave`.
"""
function integrate_trajectory!(prob, Tsave, nsaves; verbose::Bool = false)
    dt = prob.clock.dt
    save_every_steps = round(Int, Tsave / dt)
    
    nx, ny = size(prob.vars.ζ)
    zeta_data = Array{eltype(prob.vars.ζ)}(undef, nsaves, nx, ny)
    zeta_data[1, :, :] .= Array(prob.vars.ζ)

    for m in 1:(nsaves - 1)
        stepforward!(prob, save_every_steps)
        update_vorticity!(prob)
        zeta_data[m + 1, :, :] .= Array(prob.vars.ζ)
        verbose && println("saved snapshot $m at t=$(prob.clock.t)")
    end

    return zeta_data
end

"""
    warmup!(prob, ζ0, dev)

Compile the selected CPU/GPU kernels and execute the FFT plans before timing.
The caller resets the problem before the subsequent timed integration.
"""
function warmup!(prob, ζ0, dev)
    reset_problem!(prob, ζ0, dev)
    integrate_trajectory!(prob, prob.clock.dt, 2)
    synchronize_device(dev)
    return nothing
end

"""
    solve(nx, ny, Lx, Ly, ν, ζ0, F_hat, dt, Tsave,
          nsaves=101, dev=CPU(); stepper="RK4", verbose=false)

Solve one query with RK4, ETDRK4, or the local ETD1 integrator and return the
trajectory and runtime.
Use `benchmark_time_integrators` for defensible runtime measurements; a direct
timed call to this convenience function includes first-call compilation unless
the method has already been warmed up.
"""
function solve(nx, ny, Lx, Ly, ν, ζ0, F_hat, dt, Tsave,
               nsaves = 101, dev = CPU();
               stepper::AbstractString = "RK4",
               verbose::Bool = false,
               T = Float64)
    prob = build_problem(nx, ny, Lx, Ly, ν, F_hat, dt, dev; stepper, T)
    reset_problem!(prob, ζ0, dev)
    synchronize_device(dev)
    start_ns = time_ns()
    zeta_data = integrate_trajectory!(prob, Tsave, nsaves; verbose)
    synchronize_device(dev)
    runtime = (time_ns() - start_ns) * 1e-9
    return zeta_data, runtime
end


relative_l2_error(pred, ref) = norm(pred .- ref) / norm(ref)

"""
    leading_fft_work(ne, nsteps, stepper)

Return the leading FFT work for one integration. Under the manuscript's model,
one nonlinear evaluation costs `25 ne log2(ne)` flops. RK4 and ETDRK4 use four
such evaluations per step, whereas ETD1 uses one.
"""
function rhs_evaluations_per_step(stepper::AbstractString)
    return stepper == "ETD1" ? 1 : 4
end

formal_order(stepper::AbstractString) = stepper == "ETD1" ? 1 : 4

function leading_fft_work(ne, nsteps, stepper::AbstractString)
    return nsteps * 25 * rhs_evaluations_per_step(stepper) * ne * log2(ne)
end

"""
    time_step_flops(ne, stepper)

Return the modeled floating-point work per time step, including both FFTs and
pointwise operations. The count follows the operation-counting convention in
the manuscript supplement and approximates the number of stored Fourier modes
by `ne`.

For the exponential methods, one nonlinear advection/forcing evaluation costs

    C_N = 25 ne log2(ne) + 33 ne.

The diagonal viscous term is excluded from `C_N` because it is incorporated in
precomputed exponential coefficients. The ETD1 update adds `6 ne` flops, and
the three ETDRK4 stage formations and final update add `39 ne` flops. The RK4
constant is retained from the supplement.
"""
function time_step_flops(ne, stepper::AbstractString)
    fft_work = 25 * rhs_evaluations_per_step(stepper) * ne * log2(ne)
    pointwise_work = if stepper == "ETD1"
        39 * ne
    elseif stepper == "ETDRK4"
        171 * ne
    else
        184 * ne
    end
    return fft_work + pointwise_work
end

"""
    total_flops(ne, nsteps, stepper)

Return the modeled time-integration work for `nsteps` time steps. Problem
construction, FFT planning, ETD coefficient construction, and output transforms
are excluded.
"""
function total_flops(ne, nsteps, stepper::AbstractString)
    return nsteps * time_step_flops(ne, stepper) 
end

"""
    load_reference(test_index, downsample; data_dir=DEFAULT_DATA_DIR)

Load and spatially subsample one reference trajectory. The returned array stores
the forcing first, followed by vorticity at the saved times.
"""
function load_reference(test_index, downsample; data_dir = DEFAULT_DATA_DIR)
    stride = 2^downsample
    path = joinpath(data_dir, @sprintf("navier_stokes_%05d.npy", test_index))
    data = NPZ.npzread(path)
    return data[:, 1:stride:end, 1:stride:end]
end

function available_reference_indices(data_dir = DEFAULT_DATA_DIR)
    indices = Int[]
    for filename in readdir(data_dir)
        match_result = match(r"^navier_stokes_(\d+)\.npy$", filename)
        isnothing(match_result) || push!(indices, parse(Int, match_result.captures[1]))
    end
    sort!(indices)
    return indices
end

"""
    benchmark_time_integrators(; kwargs...)

Benchmark one time integrator on one device against the stored trajectories in
`../../data/navier_stokes`. By default, the final `n_trial` available files are
used. After a warmup solve, each complete `integrate_trajectory!` call is timed
once, with device synchronization immediately before and after the call.
"""
function benchmark_time_integrators(;
    stepper::AbstractString,
    device::AbstractString,
    downsample_levels = collect(0:3),
    n_trial::Int = 10,
    trial_indices = nothing,
    nt::Int = 50,
    dt_factor::Float64 = 1.0,
    base_resolution::Int = 256,
    base_dt::Float64 = 1 / 512,
    Tsave::Float64 = 1.0,
    L::Float64 = 1.0,
    ν::Float64 = 1e-4,
    T = Float64,
)
    stepper = String(stepper)
    device = lowercase(device)
    dev = device_from_name(device)

    if isnothing(trial_indices)
        available = available_reference_indices()
        trial_indices = available[(end - n_trial + 1):end]
    else
        trial_indices = collect(Int, trial_indices)
    end

    nr = length(downsample_levels)
    nsaves = nt + 1

    runtime = fill(NaN, nr, n_trial)
    accuracy = fill(NaN, nr, n_trial, nsaves)
    resolutions = Vector{Int}(undef, nr)
    time_steps = Vector{Float64}(undef, nr)
    nsteps = Vector{Int}(undef, nr)
    estimated_total_flops = Vector{Float64}(undef, nr)

    for (ir, downsample) in enumerate(downsample_levels)
        stride = 2^downsample
        n = base_resolution ÷ stride
        resolutions[ir] = n

        references = [load_reference(i, downsample) for i in trial_indices]
        F_phys = Array(references[1][1, :, :])
        zeta_refs = [Array(data[2:(nt + 2), :, :]) for data in references]
        F_hat = rfft(device_array(dev)(T.(F_phys)))

        dt = base_dt * stride * dt_factor
        steps_per_save = round(Int, Tsave / dt)
        total_steps = nt * steps_per_save
        time_steps[ir] = dt
        nsteps[ir] = total_steps
        estimated_total_flops[ir] = total_flops(n^2, total_steps, stepper)

        prob = build_problem(n, n, L, L, ν, F_hat, dt, dev; stepper, T)
        warmup!(prob, zeta_refs[1][1, :, :], dev)

        @printf("%s on %s: n=%d, dt=%.6g, steps=%d\n",
                stepper, uppercase(device), n, dt, total_steps)

        for trial in 1:n_trial
            reset_problem!(prob, zeta_refs[trial][1, :, :], dev)
            synchronize_device(dev)
            start_ns = time_ns()
            trajectory = integrate_trajectory!(prob, Tsave, nsaves)
            synchronize_device(dev)
            seconds = (time_ns() - start_ns) * 1e-9
            runtime[ir, trial] = seconds

            for j in 1:nsaves
                accuracy[ir, trial, j] = relative_l2_error(
                    trajectory[j, :, :], zeta_refs[trial][j, :, :]
                )
            end
        end
    end

    results = Dict{String, Any}(
        "runtime" => runtime,
        "accuracy" => accuracy,
        "estimated_total_flops" => estimated_total_flops,
        "resolutions" => resolutions,
        "time_steps" => time_steps,
        "nsteps" => nsteps,
        "trial_indices" => trial_indices,
        "stepper" => stepper,
        "device" => device,
        "dt_factor" => dt_factor,
    )

    return results
end

"""
    cost_accuracy_traditional_solver_helper(device, n_downsample, n_trial; kwargs...)

Compatibility entry point matching the original Navier--Stokes script. It
returns arrays with shapes `(n_downsample, n_trial, 2)` for modeled work and
runtime, and `(n_downsample, n_trial, nt+1)` for the error history.
"""
function cost_accuracy_traditional_solver_helper(
    device,
    n_downsample,
    n_trial;
    stepper = "RK4",
    nt::Int = 50,
)
    device_name = lowercase(String(device))
    levels = collect(0:(n_downsample - 1))
    data = benchmark_time_integrators(;
        downsample_levels = levels,
        n_trial,
        nt,
        stepper = String(stepper),
        device = device_name,
        dt_factor = (stepper == "RK4" || stepper == "ETDRK4") ? 1.0 : 0.25,
    )

    cost = fill(NaN, n_downsample, n_trial, 2)
    accuracy = Array(data["accuracy"])
    cost[:, :, 1] .= reshape(data["estimated_total_flops"], :, 1)
    cost[:, :, 2] .= data["runtime"]
    return cost, accuracy
end

"""
    cost_accuracy_traditional_solver(; n_downsample=4, n_trial=10, kwargs...)

Compatibility entry point matching the original script. The returned cost
array has channels `(modeled flops, CPU runtime, GPU runtime)`, and the accuracy
array has separate CPU and GPU channels.
"""
function cost_accuracy_traditional_solver(;
    n_downsample = 4,
    n_trial = 10,
    nt::Int = 50,
    stepper = "RK4",
    devices = ["gpu", "cpu"]
)
    cost = fill(NaN, n_downsample, n_trial, 3)
    accuracy = fill(NaN, n_downsample, n_trial, 2, nt + 1)

    for device in lowercase.(String.(devices))
        cost_device, accuracy_device = cost_accuracy_traditional_solver_helper(
            device, n_downsample, n_trial; stepper, nt,
        )
        cost[:, :, 1] .= cost_device[:, :, 1]
        if device == "cpu"
            cost[:, :, 2] .= cost_device[:, :, 2]
            accuracy[:, :, 1, :] .= accuracy_device
        else
            cost[:, :, 3] .= cost_device[:, :, 2]
            accuracy[:, :, 2, :] .= accuracy_device
        end
    end

    save_data = Dict("cost" => cost, "accuracy" => accuracy)
    mkpath(DEFAULT_OUTPUT_DIR)
    output_path = joinpath(
        DEFAULT_OUTPUT_DIR, "cost_accuracy_traditional_solver_data_$(stepper).npz"
    )
    NPZ.npzwrite(output_path, save_data)
    
    return cost, accuracy
end


"""Verify all three integrators against the exact Taylor--Green decay."""
function taylor_green_vortex_test(; steppers = collect(SUPPORTED_STEPPERS),
                                  dev = CPU(), nx = 64, ny = 64,
                                  ν = 1e-2, dt = 1e-3, Tfinal = 1.0)
    L = 2π
    U0 = 1.0
    k = 1
    x = LinRange(0, L, nx + 1)[1:end-1]
    y = LinRange(0, L, ny + 1)[1:end-1]
    ζ0 = [2k * U0 * sin(k * xx) * sin(k * yy) for xx in x, yy in y]
    decay = exp(-2ν * k^2 * Tfinal)
    ζ_exact = [2k * U0 * decay * sin(k * xx) * sin(k * yy) for xx in x, yy in y]
    F_hat = rfft(device_array(dev)(zeros(nx, ny)))

    errors = Dict{String, Float64}()
    for stepper in String.(steppers)
        trajectory, _ = solve(
            nx, ny, L, L, ν, ζ0, F_hat, dt, Tfinal, 2, dev; stepper
        )
        errors[stepper] = relative_l2_error(trajectory[end, :, :], ζ_exact)
        @printf("%-7s relative L2 error: %.6e\n", stepper, errors[stepper])
    end
    return errors
end

"""
    temporal_convergence_test(; steppers=["ETD1", "ETDRK4"], ...)

Verify first-order convergence of ETD1 and fourth-order convergence of ETDRK4
using a manufactured, time-dependent Taylor--Green vortex with an exact
solution.
"""
function temporal_convergence_test(;
    steppers = ["ETD1", "ETDRK4"],
    dts = [1 / 8, 1 / 16, 1 / 32, 1 / 64],
    nx::Int = 64,
    ny::Int = 64,
    Tfinal::Float64 = 1.0,
    L::Float64 = 2π,
    ν::Float64 = 1e-2,
    U0::Float64 = 1.0,
    k::Int = 1,
    frequency::Float64 = 1.0,
)
    dts = collect(Float64, dts)
    steppers = collect(String, steppers)
    nsteps = round.(Int, Tfinal ./ dts)

    x = LinRange(0, L, nx + 1)[1:end-1]
    y = LinRange(0, L, ny + 1)[1:end-1]
    ζshape = [2k * U0 * sin(k * xx) * sin(k * yy) for xx in x, yy in y]
    ζshape_hat = rfft(ζshape)
    amplitude(t) = cos(frequency * t)
    amplitude_derivative(t) = -frequency * sin(frequency * t)
    ζ0 = amplitude(0) .* ζshape
    ζexact = amplitude(Tfinal) .* ζshape

    ne = nx * ny
    results = Dict{String, Any}()
    for method in steppers
        errors = Float64[]
        work = [total_flops(ne, nt, method) for nt in nsteps]
        println("\n$method temporal convergence (formal order $(formal_order(method))):")
        for (idt, (dt, nt)) in enumerate(zip(dts, nsteps))
            function calcF!(Fh, sol, t, clock, vars, params, grid)
                coefficient = amplitude_derivative(t) + 2ν * k^2 * amplitude(t)
                @. Fh = coefficient * ζshape_hat
                return nothing
            end

            native_stepper = method == "ETD1" ? ETD1() : method
            prob = TwoDNavierStokes.Problem(
                CPU();
                nx, ny, Lx = L, Ly = L,
                ν, nν = 1, μ = 0, nμ = 0,
                dt, stepper = native_stepper,
                calcF = calcF!, T = Float64,
            )
            reset_problem!(prob, ζ0, CPU())
            trajectory = integrate_trajectory!(prob, Tfinal, 2)
            error = relative_l2_error(trajectory[end, :, :], ζexact)
            push!(errors, error)
            @printf(
                "  dt=%-10.8f steps=%4d  relative L2 error=%.6e  total flops=%.6e\n",
                dt, nt, error, work[idt],
            )
        end

        rates = [
            log(errors[i] / errors[i + 1]) / log(dts[i] / dts[i + 1])
            for i in 1:(length(dts) - 1)
        ]
        println("  observed orders: ", rates)
        results[method] = (; dts = copy(dts), nsteps = copy(nsteps),
                            errors, rates, total_flops = work)
    end

    return (;
        grid_size = (nx, ny),
        ne,
        Tfinal,
        results,
    )
end

function print_usage()
    println("Usage:")
    println("  julia spectral_navier_stokes_solver.jl taylor-green")
    println("  julia spectral_navier_stokes_solver.jl convergence")
    println("  julia spectral_navier_stokes_solver.jl benchmark STEPPER [n_trial] [nt]")
    println()
    println("To benchmark one time-step factor, include this file and call:")
    println("  benchmark_time_integrators(stepper=\"ETDRK4\", device=\"gpu\",")
    println("                             dt_factor=1.0)")
end




cost_accuracy_traditional_solver(; n_downsample = 4, n_trial = 10, nt = 50, stepper = "ETD1",)
cost_accuracy_traditional_solver(; n_downsample = 4, n_trial = 10, nt = 50, stepper = "ETDRK4",)
     
