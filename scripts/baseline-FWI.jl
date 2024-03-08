## A 2D compass example
using DrWatson
@quickactivate "TL-FWPI"
using Pkg; Pkg.instantiate();

nthreads = try
    # Slurm
    parse(Int, ENV["SLURM_CPUS_ON_NODE"])
    using ThreadPinning
    pinthreads(:cores)
catch e
    # Desktop
    Sys.CPU_THREADS
end
using LinearAlgebra
BLAS.set_num_threads(nthreads)

include(srcdir("dummy_src_file.jl"))
using JUDI
dummy_JUDI_operation()
using PyPlot
using Flux
using LineSearches
using JLD2
using Statistics
using Images
using Random
Random.seed!(2023)

matplotlib.use("agg")

sim_name = "fwi"
exp_name = "baseline"

mkpath(datadir())
mkpath(plotsdir())

## load compass
JLD2.@load datadir("image2023_v_rho.jld2") v rho
vp = deepcopy(v)
n = size(vp)
d = (6f0, 6f0)
o = (0f0, 0f0)
model = Model(n, d, o, (1f0 ./ vp).^2f0; nb = 80)   # wave model

timeS = timeR = 3600f0               # recording time
dtS = dtR = 4f0                     # recording time sampling rate
ntS = Int(floor(timeS/dtS))+1       # time samples
ntR = Int(floor(timeR/dtR))+1       # source time samples
idx_wb = minimum(find_water_bottom(vp.-minimum(vp)))
nsrc = 128
nrec = n[end]
extentx = (n[1]-1)*d[1];
extentz = (n[2]-1)*d[2];

mode = "both"
if mode == "reflection"
    xsrc = convertToCell(Float32.(ContJitter(extentx, nsrc)))
    zsrc = convertToCell(range(10f0,stop=10f0,length=nsrc))
    xrec = range(d[1],stop=(n[1]-1)*d[1],length=nrec)
    zrec = range((idx_wb-1)*d[2]-2f0,stop=(idx_wb-1)*d[2]-2f0,length=nrec)
elseif mode == "transmission"
    xsrc = convertToCell(range(d[1],stop=d[1],length=nsrc))
    zsrc = convertToCell(Float32.(ContJitter(extentz, nsrc)))
    xrec = range((n[1]-1)*d[1],stop=(n[1]-1)*d[1], length=nrec)
    zrec = range((idx_wb-1)*d[2]+10f0,stop=(n[2]-1)*d[2],length=nrec)
else
    # source locations -- half at the left hand side of the model, half on top
    xsrc = convertToCell(vcat(Float32.(ContJitter(extentx, div(nsrc,2))),range(d[1],stop=d[1],length=div(nsrc,2))))
    zsrc = convertToCell(vcat(range(10f0,stop=10f0,length=div(nsrc,2)),Float32.(ContJitter(extentz, div(nsrc,2)))))
    xrec = vcat(range((n[1]-1)*d[1],stop=(n[1]-1)*d[1], length=div(nrec,2)),range(d[1],stop=(n[1]-1)*d[1],length=div(nrec,2)))
    zrec = vcat(range((idx_wb-1)*d[2]+10f0,stop=(n[2]-1)*d[2],length=div(nrec,2)),range(10f0,stop=10f0,length=div(nrec,2)))
end

ysrc = convertToCell(range(0f0,stop=0f0,length=nsrc))
yrec = 0f0

# set up src/rec geometry
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

# set up source
f0 = 0.02f0     # kHz
wavelet = ricker_wavelet(timeS, dtS, f0)
q = judiVector(srcGeometry, wavelet)

# set up simulation operators
F = judiModeling(model, srcGeometry, recGeometry) # acoustic wave equation solver

d_obs = F * q

wb = maximum(find_water_bottom(vp.-minimum(vp)))
v0 = deepcopy(vp)
v0[:,wb+1:end] .= 1f0./Float32.(imfilter(1f0./vp[:,wb+1:end], Kernel.gaussian(10)))
m0 = 1f0./v0.^2f0
m_init = deepcopy(m0)
model0 = Model(n, d, o, m0; nb = 80)
proj(x::AbstractVecOrMat{T}; upper=T(1f0./1.48f0.^2f0), lower=T(1f0./5f0.^2f0)) where T = max.(min.(x,T(upper)),T(lower))
niterations = 500
nsrc_all = q.nsrc
nssample = 8
fhistory = zeros(niterations)
F0 = judiModeling(deepcopy(model0), q.geometry, d_obs.geometry)
ls = BackTracking(order=3, iterations=10)

for j=1:niterations

    Base.flush(Base.stdout)

    ### subsample sources
    rand_ns = jitter(nsrc_all, nssample)                          # select random source idx for each vintage
    fval, gradient = fwi_objective(model0, q[rand_ns], d_obs[rand_ns])
    p = -gradient/norm(gradient, Inf)
    fhistory[j] = fval
    println("Inversion iteration no: ",j,"; function value: ", fhistory[j])

    # linesearch
    function f_(α)
        F0.model.m .= proj(model0.m .+ α * p)
        misfit = .5*norm(F0[rand_ns]*q[rand_ns] - d_obs[rand_ns])^2
        @show α, misfit
        return misfit
    end
    step, fval = ls(f_, 1f-1, fval, dot(gradient, p))
    model0.m .= proj(model0.m .+ step .* p)

    ### save intermediate results
    save_dict = @strdict mode j nssample f0 model0 gradient niterations nsrc nrec n d fhistory
    @tagsave(
        joinpath(datadir(sim_name, exp_name), savename(save_dict, "jld2"; digits=6)),
        save_dict;
        safe=true
    )

    ## save figure
    fig_name = @strdict mode j nssample f0 model0 niterations nsrc nrec n d fhistory

    ## loss
    fig = figure(figsize=(20,12));
    plot(fhistory[1:j]);title("loss=$(fhistory[j])");
    suptitle("FWI at iter $(j)")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_loss.png"), fig);
    close(fig)

    ## model predicting
    cmin, cmax = extrema(1f0./vp.^2f0)
    fig = figure(figsize=(20,12));
    subplot(1,4,1);
    imshow(m_init', vmin=cmin, vmax=cmax);
    title("initial")
    subplot(1,4,2);
    imshow(1f0./vp'.^2f0, vmin=cmin, vmax=cmax);
    title("true")
    subplot(1,4,3);
    imshow(model0.m.data', vmin=cmin, vmax=cmax);
    title("inverted")
    subplot(1,4,4);
    imshow(5*abs.(model0.m.data-1f0./vp.^2f0)', cmap="magma");
    title("5X diff")
    suptitle("FWI at iter $(j)")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_squaredslowness.png"), fig);
    close(fig)

end
