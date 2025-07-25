struct InterpolatedExactSimpleCorrection <: AbstractCorrection end

function setup_correction(::InterpolatedExactSimpleCorrection, d::Dict)
    return analytical_exact_simple_correction(d[:η], d[:k], d[:ts], d[:totalparity])
end
function analytical_exact_simple_correction(η, k, ts, totalparity; opt_kwargs...)
    results = Float64[]
    for t in ts
        initial = 0.0
        result = find_zero_energy_from_analytics(η, k, t, initial, totalparity; opt_kwargs...)
        push!(results, result)
    end
    return SimpleCorrection(linear_interpolation(ts, results, extrapolation_bc=Periodic()))
end

find_zero_energy_from_analytics(η::Tuple, k, t, initial, totalparity; kwargs...) = find_zero_energy_from_analytics(effective_η(η), k, t, initial, totalparity; kwargs...)
function find_zero_energy_from_analytics(η, k, t, initial, totalparity; atol=1e-15, rtol=0.0, kwargs...)
    λ = try
        find_zero(λ -> analytic_parameters(λ, η, k, t).ε - λ, initial; atol, rtol, kwargs...)
    catch
        find_zero(λ -> analytic_parameters(λ, η, k, t).ε - λ, initial; atol, rtol, verbose=true, kwargs...)
    end
    return λ
end


function analytic_energy_spectrum(λ, η, k, t, totalparity)
    (; ε, Δtilde) = analytic_parameters(λ, η, k, t)
    es = [0, -ε + λ, -ε + Δtilde, Δtilde + λ]
    2 * sort(es .- sum(es) / 4)
end

function effective_η(ηs::Tuple)
    return sqrt(ηs[1] * sqrt(ηs[2] * ηs[3]))
end

analytic_parameters(λ, η::Tuple, k, t) = analytic_parameters(λ, effective_η(η), k, t)
function analytic_parameters(λ, η, k, t)
    ρs = get_rhos(k, t) 
    ρ23 = sqrt(ρs[2]^2 + ρs[3]^2)
    θ = atan(ρ23 / ρs[1])
    ϕ = atan(ρs[3] / ρs[2])
    λtilde = λ * sin(θ) + η * cos(θ) * sin(θ)
    ηtilde = η * sin(θ)^2 - λ * cos(θ)
    θ_μ = -1 / 2 * atan(2 * λtilde * ηtilde, 1 + λtilde^2 - ηtilde^2)
    θ_α = atan(ηtilde * tan(θ_μ) - λtilde)

    ν, μ = sincos(θ_μ)
    β, α = sincos(θ_α)
    Δtilde = (α * μ + ηtilde * β * ν - λtilde * β * μ)

    ε = β * ν + λtilde * α * ν + ηtilde * α * μ # = ηtilde * α / μ. 

    return (; ηtilde, λtilde, μ, α, β, ν, θ_α, θ_μ, Δtilde, ε, θ, ϕ, η, λ)
end

analytical_components_middle_of_protocol(d::Dict) = analytical_components_middle_of_protocol(d[:η], d[:totalparity])
analytical_components_middle_of_protocol(η::Tuple, totalparity) = analytical_components_middle_of_protocol(effective_η(η), totalparity)
function analytical_components_middle_of_protocol(η, totalparity)
    λ = η / sqrt(1 + η^2)
    d = sqrt(1 + η^2 + η^4)
    α = 1 / d
    ν = -η^2 / d
    μ = sqrt(1 + η^2) / d
    β = -η * sqrt(1 + η^2) / d
    θ_α = atan(β, α)
    θ_μ = atan(ν, μ)
    return (; μ, α, β, ν, θ_α, θ_μ, λ, η)
end

function analytical_components_middle_of_protocol_old(η, totalparity)
    λ = η / sqrt(1 + η^2)
    θ_μ = -1 / 2 * atan(2 * λ * η / (1 + λ^2 - η^2))
    ν, μ = sincos(θ_μ)
    θ_α = -1 * atan(-η * tan(θ_μ) + λ)
    β, α = sincos(θ_α)
    return (; μ, α, β, ν, θ_α, θ_μ, λ, η)
end


@testitem "Analytic spectrum" begin
    using StaticArrays, LinearAlgebra
    mtype, vtype = SMatrix{4,4,ComplexF64}, SVector{4}

    T = 10
    k = 1
    λ = 0.2
    corr = SimpleCorrection(λ)
    η = 0.5
    gapscaling = t -> 1
    ts = range(0, 2, 5)
    for totalparity in (1, -1)
        P = parity_operators(totalparity, mtype)
        p = (η, k, gapscaling, corr, P, totalparity)
        spectrum = stack(map(t -> eigvals(MajoranaBraiding.ham_with_corrections(p, t)), ts))'
        analytic_spectrum = stack(map(t -> MajoranaBraiding.analytic_energy_spectrum(λ, η, k, t, totalparity), ts))'
        @test spectrum ≈ analytic_spectrum
    end
    param_dict = Dict(
        :η => 0.7, #Majorana overlaps. Number or triplet of numbers
        :T => 1e4, #Maximum time
        :k => 2e1, #Determines the slope of the ramp
        :steps => 1000, #Number of timesteps for interpolations
        :correction => NoCorrection(), #Different corrections are available. This is the most relevant one for the paper
        :interpolate_corrected_hamiltonian => false, #Creating an interpolated Hamiltonian might speed things up
        :initial => I, #Initial state. Use U0 for the identity matrix.
        :totalparity => 1
    )
    prob = setup_problem(param_dict)
    ts = range(0, 2, 5)
    λ = 0
    spectrum = stack(map(t -> eigvals(prob[:H](prob[:p], t) / prob[:T]), ts))'
    analytic_spectrum = stack(map(t -> MajoranaBraiding.analytic_energy_spectrum(λ, prob[:η], prob[:k], t, prob[:totalparity]), ts))'
    @test norm(spectrum - analytic_spectrum) < 1e-10

    param_dict[:totalparity] = -1
    prob = setup_problem(param_dict)
    spectrum = stack(map(t -> eigvals(prob[:H](prob[:p], t) / prob[:T]), ts))'
    analytic_spectrum = stack(map(t -> MajoranaBraiding.analytic_energy_spectrum(λ, prob[:η], prob[:k], t, prob[:totalparity]), ts))'
    @test norm(spectrum - analytic_spectrum) < 1e-10
end