# Examples
This section contains different examples that describes the usage of [GibbsSampler.jl](https://github.com/efmanu/GibbsSampler.jl) package.

## Different MCMC Samplers for parameter sampling

The [GibbsSampler.jl](https://github.com/efmanu/GibbsSampler.jl) allows the use of **MH**, **HMC**, **NUTS** MCMC samplers with the help of [AdvancedMH.jl](https://github.com/TuringLang/AdvancedMH.jl) and [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl) package. The `GibbsSampler.gibbs(...)` function has an input corresponds to `alg`, which decides the MCMC sampler availble based on structs defined by the package (Eg: `MH()`, `adHMC()`, `adNUTS()` etc.). This algorithm need to to be mapped to parameter groups using another dictionary input `sample_alg`. The key of `sample_alg` represents the parameter group and the value vector contains the index of the sampling algorithm defined in `alg` with proposal distribution.

### Use of AdvancedMH as MCMC sampler
The `MH()` struct defined with [GibbsSampler.jl](https://github.com/efmanu/GibbsSampler.jl) package is used to select MCMC sampler for each parameter in Gibbs sampling. 


#### Example
```julia
#use packages
using GibbsSampler
using Distributions

#define MCMC samplers
alg = [MH(), adHMC()]

#define sample_alg parameter
sample_alg = Dict(
	:n_grp => 2,
	1 => Dict(
		:type => :ind,
		:n_vars => 2,
		1 => Dict(
			:proposal => MvNormal(zeros(2),1.0),
			:n_eles => 2,
			:alg => 1
		),
		2 => Dict(
			:proposal => Normal(0.0,1.0),
			:n_eles => 1,
			:alg => 1
		)
	),
	2 => Dict(
		:type => :dep,
		:n_vars => 2,
		:alg => 2,
		1 => Dict(
			:proposal => MvNormal(zeros(3),1.0),
			:n_eles => 3
		),
		2 => Dict(
			:proposal => Normal(0.0,1.0),
			:n_eles => 1
		)
	)
)
#define prior distribution
prior = [MvNormal([1.0,2.0],1.0),Normal(2.0,1.0), MvNormal([2.0,4.0,3.0],1.0),Normal(-1.0,1.0)]

#define logjoint function
logJoint(params) = sum(logpdf.(prior, params))

#sample
chn = gibbs(alg, sample_alg, logJoint, itr = 10000, chain_type = :mcmcchain)
```

