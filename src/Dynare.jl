module Dynare

using ExtendedDates
using Logging
using Pkg

Base.@kwdef struct CommandLineOptions
    compilemodule::Bool = true
end

using LinearRationalExpectations
const LRE = LinearRationalExpectations

using KalmanFilterTools

include("utils.jl")
include("dynare_functions.jl")
include("dynare_containers.jl")
include("accessors.jl")
export irf, simulation
include("model.jl")
export get_abc, get_de
include("symboltable.jl")
include("data.jl")
include("distributions/distribution_parameters.jl")
include("estimation/parse_prior.jl")
include("initialization.jl")
include("deterministic_trends.jl")
include("DynareParser.jl")
export parser
include("DynarePreprocessor.jl")
export dynare_preprocess
include("steady_state/SteadyState.jl")
export steady_state!
include("dynare_table.jl")
export round
include("reporting/report.jl")
include("graphics.jl")
include("filters/kalman/kalman.jl")
include("optimal_policy.jl")
include("perturbations.jl")
include("perfectforesight/perfectforesight.jl")
include("estimation/priorprediction.jl")
include("simulations.jl")
include("nonlinear/NLsolve.jl")
using .NLsolve
include("estimation/estimation.jl")
export mh_estimation

export @dynare, @precompile_model, @dynare1

macro dynare(modfile_arg::String, args...)
    @info "Dynare version: $(module_version(Dynare))"
    modname = get_modname(modfile_arg)
    @info "$(now()): Starting @dynare $modfile_arg"
    arglist = []
    compilemodule = true
    preprocessing = true
    for (i, a) in enumerate(args)
        if a == "nocompile"
            compilemodule = false
        elseif a == "nopreprocessing"
            preprocessing = false
        else
            push!(arglist, a)
        end
    end
    if preprocessing
        modfilename = modname * ".mod"
        dynare_preprocess(modfilename, arglist)
    end
    @info "$(now()): End of preprocessing"
    options = CommandLineOptions(compilemodule)
    context = parser(modname, options)
    return context
end

function get_modname(modfilename::String)
    if occursin(r"\.mod$", modfilename)
        modname::String = modfilename[1:length(modfilename)-4]
    else
        modname = modfilename
    end
    return modname
end

macro precompile_model(modfile_arg::String, args...)
    modname = get_modname(modfile_arg)
    @info "$(now()): Starting @dynare $modfile_arg"
    arglist = []
    compilemodule = true
    preprocessing = true
    for (i, a) in enumerate(args)
        if a == "nocompile"
            compilemodule = false
        elseif a == "nopreprocessing"
            preprocessing = false
        else
            push!(arglist, a)
        end
    end
    modfilename = modname * ".mod"
    dynare_preprocess(modfilename, arglist)
    update_package(modname)
end

macro dynare1(modfile_arg::String, args...)
    @info "Dynare version: $(module_version(Dynare))"
    modname = get_modname(modfile_arg)
    modeldir = joinpath(modname, "model/julia")
    make_module(modeldir)
    !(modeldir in LOAD_PATH) && push!(LOAD_PATH, modeldir)
    @eval using DynareFunctions
    compilemodule = true
    options = CommandLineOptions(compilemodule)
    context = parser(modname, options)
    return context
end

function make_module(directory::String)
    files = readdir(directory)
    if !("DynareFunctions.jl" in files)
        open(joinpath(directory, "DynareFunctions.jl"), "w") do io
            write(io, "module DynareFunctions\n")
            for f in files
                if f != "DynareFunctions.jl"
                    write(io, "include(\"$f\")\n")
                end
            end
            write(io, "end\n")
        end
    end
end

function update_package(modfilename)
    functiondir = joinpath(modfilename, "model/julia")
    projectdir = joinpath(functiondir, "project")
    if !isdir(projectdir)
        Pkg.generate(projectdir)
        Pkg.develop(path=projectdir)
        for f in readdir(functiondir)
            ff = split(f, ".")
            if ff[end] == "jl"
                cp(joinpath(functiondir,f), joinpath(projectdir, "src", f), force=true)
            end
        end
    else
        for f in readdir(functiondir)
            if mtime(f) > mtime(join(projectdir, "src", f))
                ff = split(f, ".")
                if ff[end] == "jl"
                    cp(joinpath(functiondir,f), joinpath(projectdir, "src", f), force=true)
                end
            end
        end
    end
end

#include("precompile_Dynare.jl")
#_precompile_()
end # module

