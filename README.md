<h1 align="center">Time-lapse full-waveform permeability inversion: a feasibility study</h1>

[![][license-img]][license-status] [![][zenodo-img]][zenodo-status]

Code to reproduce results in Ziyi Yin, Mathias Louboutin, Olav Møyner, Felix J. Herrmann, "[Time-lapse full-waveform permeability inversion: a feasibility study](https://arxiv.org/abs/2403.04083)". DOI: 10.48550/arXiv.2403.04083

## Software descriptions

All of the software packages used in this paper are fully *open source, scalable, interoperable, and differentiable*. The readers are welcome to learn about our software design principles from [this open-access article](https://library.seg.org/doi/10.1190/tle42070474.1).

#### Wave modeling

We use [JUDI.jl](https://github.com/slimgroup/JUDI.jl) for wave modeling and inversion, which calls the highly optimized propagators of [Devito](https://www.devitoproject.org/).

#### Multiphase flow

We use [JutulDarcyRules.jl] to solve the multiphase flow equations, which calls the high-performant and auto-differentiable numerical solvers in [Jutul.jl] and [JutulDarcy.jl]. [JutulDarcyRules.jl] is designed to interoperate these two packages with other Julia packages in the Julia AD ecosystem via [ChainRules.jl].

## Installation

First, install [Julia](https://julialang.org/) and [Python](https://www.python.org/). The scripts will contain package installation commands at the beginning so the packages used in the experiments will be automatically installed.

## Scripts

[case-1.jl](scripts/case-1.jl) runs end-to-end permeability inversion with a homogeneous initial permeability model.

[case-2.jl](scripts/case-2.jl) runs end-to-end permeability inversion with a distorted initial permeability model.

[baseline-FWI.jl](scripts/baseline-FWI.jl) runs full-waveform inversion to estimate the baseline brine-filled velocity model.

[case-3.jl](scripts/case-3.jl) uses the inverted baseline brine-filled velocity model obtained above to run end-to-end permeability inversion.

## LICENSE

The software used in this repository can be modified and redistributed according to [MIT license](LICENSE).

## Reference

If you use our software for your research, we appreciate it if you cite us following the bibtex in [CITATION.bib](CITATION.bib).

## Authors

This repository is written by [Ziyi Yin] from the [Seismic Laboratory for Imaging and Modeling] (SLIM) at the Georgia Institute of Technology.

If you have any question, we welcome your contributions to our software by opening issue or pull request.

SLIM Group @ Georgia Institute of Technology, [https://slim.gatech.edu](https://slim.gatech.edu/).      
SLIM public GitHub account, [https://github.com/slimgroup](https://github.com/slimgroup).    

[license-status]:LICENSE
[license-img]:http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat?style=plastic
[Seismic Laboratory for Imaging and Modeling]:https://slim.gatech.edu/
[Ziyi Yin]:https://ziyiyin97.github.io/
[Jutul.jl]:https://github.com/sintefmath/Jutul.jl
[JutulDarcy.jl]:https://github.com/sintefmath/JutulDarcy.jl
[JutulDarcyRules.jl]:https://github.com/slimgroup/JutulDarcyRules.jl
[ChainRules.jl]:https://github.com/JuliaDiff/ChainRules.jl
[zenodo-status]:https://doi.org/10.5281/zenodo.10910283
[zenodo-img]:https://zenodo.org/badge/DOI/10.5281/zenodo.10910283.svg?style=plastic