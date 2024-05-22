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
using JutulDarcyRules
using PyPlot
using Flux
using LineSearches
using JLD2
using Statistics
using Images
using Random
using Polynomials
import DrWatson: _wsave
_wsave(s, fig::Figure) = fig.savefig(s, bbox_inches="tight", dpi=300)
Random.seed!(2023)

matplotlib.use("agg")

sim_name = "end2end-inv"
exp_name = "2024-05-13-poro"

mkpath(datadir())
mkpath(plotsdir())

## load compass
JLD2.@load datadir("image2023_v_rho.jld2") v rho
vp = deepcopy(v)
n = size(vp)
d = (6f0, 6f0)

cut_area = [1, n[1], 182, n[end]]
v = Float64.(vp[cut_area[1]:cut_area[2], cut_area[3]:cut_area[4]])
factor_x = 1
factor_z = 1
factor = (factor_x,factor_z)
d = (6., 6.)
h = 181 * d[end]
v = 1.0./downsample(1.0./v, factor)
ds = Float64.(d) .* factor;

ns = (size(v,1), 1, size(v,2))
ds = (ds[1], ds[1]*ns[1], ds[2])

Kh = VtoK.(v);
K = Float64.(Kh * md);

# set up jutul model
kvoverkh = 0.1
α = 6.0
ϕ = Ktoϕ.(Kh; α=α)
model = jutulModel(ns, ds, vec(padϕ(ϕ)), K1to3(K; kvoverkh=kvoverkh), h)

## simulation time steppings
tstep = 365.25 * 5 * ones(5)
tot_time = sum(tstep)

## injection & production
inj_loc = (128 * 3 / factor_x, 1, (ns[end]-20) * 2 / factor_z) .* ds
pore_volumes = sum(ϕ[2:end-1,1:end-1] .* (v[2:end-1,1:end-1].>3.5)) * prod(ds)
irate = 0.2 * pore_volumes / tot_time / 24 / 60 / 60
f = jutulVWell(irate, (inj_loc[1], inj_loc[2]); startz = 46 * 2 / factor_z * ds[end], endz = 48 * 2 / factor_z * ds[end])
## set up modeling operator
S = jutulModeling(model, tstep)

## simulation
mesh = CartesianMesh(model)
T(x) = log.(KtoTrans(mesh, K1to3(exp.(x); kvoverkh=kvoverkh)))

logK = log.(K)

@time state = S(T(log.(ϕtoK.(ϕ;α=α)*md)),vec(padϕ(ϕ)),f)

### observed states
nv = length(tstep)
function O(state::AbstractVector)
    full_his = Float32.(reshape(state[1:nv*prod(ns)], ns[1], ns[end], nv))
    return [full_his[:,:,i] for i = 1:nv]
end
sw_true = O(state)

known_idx = [findlast(v[i,:].<=3.5) for i = 1:size(Kh,1)]
mask = Vector{Matrix{Float32}}(undef, nv)
for i = 1:nv

    energy_i = [norm(sw_true[i][ix,:]) for ix in axes(sw_true[i],1)]
    first_i = findfirst(energy_i.>0)
    end_i = findlast(energy_i.>0)
    mask_i = zeros(Float32, size(sw_true[i]))
    mask_i[max(1, first_i-5):min(end_i+5, size(sw_true[i],1)), :] .= 1f0
    mask_i = Float32.(imfilter(mask_i, Kernel.gaussian(3)))
    for j in axes(mask_i, 1)
        mask_i[j,1:known_idx[j]] .= 0f0
    end
    mask[i] = mask_i

end
M(sw::Vector{Matrix{Float32}}) = [sw[i] .* mask[i] for i = 1:length(sw)]

### pad co2 back to normal size
pad(c::Matrix{Float32}) =
    hcat(zeros(Float32, n[1], cut_area[3]-1),
    vcat(zeros(Float32, cut_area[1]-1, factor[2] * size(c,2)), repeat(c, inner=factor), zeros(Float32, n[1]-cut_area[2], factor[2] * size(c,2))))
pad(c::Vector{Matrix{Float32}}) = [pad(c[i]) for i = 1:nv]
padϕ_(c::Matrix) =
    hcat(1f-3*ones(Float32, n[1], cut_area[3]-1),
    vcat(1f-3*ones(Float32, cut_area[1]-1, factor[2] * size(c,2)), repeat(Float32.(c), inner=factor), 1f-3*ones(Float32, n[1]-cut_area[2], factor[2] * size(c,2))))

sw_pad = pad(sw_true)

# set up rock physics
bulk_min = Float32.(1f9 .* rho .* 5f0/9f0 .* vp.^2f0 .* 2f0 .* padϕ_(ϕ)/mean(ϕ[v.>3.5]))
R(c::Vector{Matrix{Float32}}) = Patchy(c,1f3*vp,1f3*rho,0.25f0 * ones(Float32,n); bulk_min = 5f10)[1]/1f3
R(c::Vector{Matrix{Float32}}, phi::Matrix{Float32}) = Patchy(c,1f3*vp,1f3*rho,phi)[1]/1f3
vps = R(sw_pad, padϕ_(ϕ))   # time-varying vp

##### Wave equation
o = (0f0, 0f0)          # origin

nsrc = 32       # num of sources
nrec = 600      # num of receivers

models = [Model(n, d, o, (1f0 ./ vps[i]).^2f0; nb = 80) for i = 1:nv]   # wave model

timeS = timeR = 3600f0               # recording time
dtS = dtR = 4f0                     # recording time sampling rate
ntS = Int(floor(timeS/dtS))+1       # time samples
ntR = Int(floor(timeR/dtR))+1       # source time samples
idx_wb = minimum(find_water_bottom(vp.-minimum(vp)))

extentx = (n[1]-1)*d[1];
extentz = (n[2]-1)*d[2];

mode = "transmission"
if mode == "reflection"
    xsrc = [convertToCell(Float32.(ContJitter(extentx, nsrc))) for i=1:nv]
    zsrc = [convertToCell(range(10f0,stop=10f0,length=nsrc)) for i=1:nv]
    xrec = range(d[1],stop=(n[1]-1)*d[1],length=nrec)
    zrec = range((idx_wb-1)*d[2]-2f0,stop=(idx_wb-1)*d[2]-2f0,length=nrec)
elseif mode == "transmission"
    xsrc = [convertToCell(range(d[1],stop=d[1],length=nsrc)) for i=1:nv]
    zsrc = [convertToCell(Float32.(ContJitter(extentz, nsrc))) for i=1:nv]
    xrec = range((n[1]-1)*d[1],stop=(n[1]-1)*d[1], length=nrec)
    zrec = range((idx_wb-1)*d[2]+10f0,stop=(n[2]-1)*d[2],length=nrec)
else
    # source locations -- half at the left hand side of the model, half on top
    xsrc = [convertToCell(vcat(Float32.(ContJitter(extentx, div(nsrc,2))),range(d[1],stop=d[1],length=div(nsrc,2)))) for i = 1:nv]
    zsrc = [convertToCell(vcat(range(10f0,stop=10f0,length=div(nsrc,2)),Float32.(ContJitter(extentz, div(nsrc,2))))) for i = 1:nv]
    xrec = vcat(range((n[1]-1)*d[1],stop=(n[1]-1)*d[1], length=div(nrec,2)),range(d[1],stop=(n[1]-1)*d[1],length=div(nrec,2)))
    zrec = vcat(range((idx_wb-1)*d[2]+10f0,stop=(n[2]-1)*d[2],length=div(nrec,2)),range(10f0,stop=10f0,length=div(nrec,2)))
end

ysrc = convertToCell(range(0f0,stop=0f0,length=nsrc))
yrec = 0f0

# set up src/rec geometry
srcGeometry = [Geometry(xsrc[i], ysrc, zsrc[i]; dt=dtS, t=timeS) for i = 1:nv]
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

# set up source
f0 = 0.02f0     # kHz
wavelet = ricker_wavelet(timeS, dtS, f0)
q = [judiVector(srcGeometry[i], wavelet) for i = 1:nv]

# set up simulation operators
Fs = [judiModeling(models[i], srcGeometry[i], recGeometry) for i = 1:nv] # acoustic wave equation solver

## wave physics
function F(v::Vector{Matrix{Float32}})
    m = [vec(1f0./v[i]).^2f0 for i = 1:nv]
    return [Fs[i](m[i], q[i]) for i = 1:nv]
end

### generate/load data
global d_obs = [Fs[i]*q[i] for i = 1:nv]
snr = 8f0

# Generate band-limited noise
noise = deepcopy(d_obs)
for k = 1:nv
    for l = 1:nsrc
        noise[k].data[l] = randn(Float32, size(d_obs[k].data[l]))
        noise[k].data[l] = real.(ifft(fft(noise[k].data[l]).*fft(q[k].data[1])))
    end
end

# Scale noise based on SNR
noise = noise/norm(noise) * norm(d_obs) * 10f0^(-snr/20f0)
d_obs = d_obs + noise

ls = BackTracking(order=3, iterations=10)

# Main loop
niterations = 200
nssample = 4
fhistory = zeros(niterations)

#### inversion
ϕ0 = deepcopy(ϕ)
ϕ_init_val = 0.125
ϕ0[v.>3.5] .= ϕ_init_val
ϕ0_init = deepcopy(ϕ0)
dϕ = 0 .* ϕ
ϕ_init = deepcopy(ϕ0)

logK0 = log.(ϕtoK.(ϕ0;α=α)*md)
logK_init = deepcopy(logK0)
@time y_init = box_co2(M(O(S(T(log.(ϕtoK.(ϕ0;α=α)*md)),vec(padϕ(ϕ0)),f))));

for j=1:niterations

    Base.flush(Base.stdout)

    ### subsample sources
    rand_ns = [jitter(nsrc, nssample) for i = 1:nv]                             # select random source idx for each vintage
    q_sub = [q[i][rand_ns[i]] for i = 1:nv]                                        # set-up source
    F_sub = [Fs[i][rand_ns[i]] for i = 1:nv]                                 # set-up wave modeling operator
    dobs = [d_obs[i][rand_ns[i]] for i = 1:nv]                                  # subsampled seismic dataset from the selected sources
    function F(v::Vector{Matrix{Float32}})
        m = [vec(1f0./v[i]).^2f0 for i = 1:nv]
        return [F_sub[i](m[i], q_sub[i]) for i = 1:nv]
    end

    # objective function for inversion
    function obj(dϕ)
        global ϕ_j = box_ϕ(ϕ0+mask[end].*dϕ)
        global c_j = box_co2(M(O(S(T(log.(ϕtoK.(ϕ_j;α=α)*md)),vec(padϕ(ϕ_j)),f))));
        global dpred_j = F(box_v(R(pad(c_j),padϕ_(ϕ_j))))
        fval = .5f0 * norm(dpred_j-dobs)^2f0/nssample/nv
        @show fval
        return fval
    end
    ## AD by Flux
    @time fval, gs = Flux.withgradient(() -> obj(dϕ), Flux.params(dϕ))

    fhistory[j] = fval
    println("Inversion iteration no: ",j,"; function value: ", fhistory[j])
    g = gs[dϕ]
    p = -g/norm(g, Inf)

    # linesearch
    function f_(α)
        try
            misfit = obj(dϕ + α * p)
            @show α, misfit
            return misfit
        catch e
            @show e
            return Inf32
        end
        return Inf32
    end

    #step, fval = ls(f_, 2e-1, fval, dot(g, p))
    step = 0.1f0
    fval  = f_(step)
    global dϕ = dϕ + step * p

    ### save intermediate results
    save_dict = @strdict ϕ_init_val mode j nssample f0 dϕ ϕ0 g niterations nv nsrc nrec nv cut_area tstep factor n d fhistory mask kvoverkh α
    @tagsave(
        joinpath(datadir(sim_name, exp_name), savename(save_dict, "jld2"; digits=6)),
        save_dict;
        safe=true
    )

    ## save figure
    fig_name = @strdict ϕ_init_val mode j nssample f0 dϕ ϕ0 niterations nv nsrc nrec nv cut_area tstep factor n d fhistory mask kvoverkh α

    ## compute true and plot
    SNR = -2f1 * log10(norm(ϕ-ϕ_j)/norm(ϕ))
    fig = figure(figsize=(20,12));
    subplot(2,2,1);
    imshow(ϕ_j', vmin=0, vmax=maximum(ϕ));title("inversion, SNR=$(SNR)");colorbar();
    subplot(2,2,2);
    imshow(ϕ', vmin=0, vmax=maximum(ϕ));title("GT permeability");colorbar();
    subplot(2,2,3);
    imshow(ϕ_init', vmin=0, vmax=maximum(ϕ));title("initial permeability");colorbar();
    subplot(2,2,4);
    imshow(ϕ_j'-ϕ_init', vmin=-0.5*maximum(ϕ), vmax=0.5*maximum(ϕ), cmap="magma");title("updated");colorbar();
    suptitle("End-to-end Inversion at iter $(j)")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_ϕ.png"), fig);
    close(fig)

    ## loss
    fig = figure(figsize=(20,12));
    plot(fhistory[1:j]);title("loss=$(fhistory[j])");
    suptitle("End-to-end Inversion at iter $(j)")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_loss.png"), fig);
    close(fig)

    ## data fitting
    fig = figure(figsize=(20,12));
    for i = 1:5
        subplot(4,5,i);
        imshow(y_init[i]', vmin=0, vmax=1);
        title("initial prediction at snapshot $(i)")
        subplot(4,5,i+5);
        imshow(sw_true[i]', vmin=0, vmax=1);
        title("true at snapshot $(i)")
        subplot(4,5,i+10);
        imshow(c_j[i]', vmin=0, vmax=1);
        title("predict at snapshot $(i)")
        subplot(4,5,i+15);
        imshow(5*(sw_true[i]'-c_j[i]'), vmin=-1, vmax=1, cmap="magma");
        title("5X diff at snapshot $(i)")
    end
    suptitle("End-to-end Inversion at iter $(j)")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_saturation.png"), fig);
    close(fig)

end
