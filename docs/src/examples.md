# Examples
This section contains different examples that describes the usage of [GibbsSampler.jl](https://github.com/efmanu/GibbsSampler.jl) package.

## Different MCMC Samplers for parameter sampling

The [GibbsSampler.jl](https://github.com/efmanu/GibbsSampler.jl) allows the use of **MH**, **HMC**, **NUTS** MCMC samplers with the help of [AdvancedMH.jl](https://github.com/TuringLang/AdvancedMH.jl) and [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl) package. The `GibbsSampler.gibbs(...)` function has an input corresponds to `alg`, which decides the MCMC sampler availble based on structs defined by the package (Eg: `MH()`, `adHMC()`, `adNUTS()` etc.). This algorithm need to to be mapped to parameter groups using another dictionary input `sample_alg`. The key of `sample_alg` represents the parameter group and the value vector contains the index of the sampling algorithm defined in `alg` with proposal distribution.

### Use of AdvancedMH as MCMC sampler
The `MH()` struct defined with [GibbsSampler.jl](https://github.com/efmanu/GibbsSampler.jl) package is used to select MCMC sampler for each parameter in Gibbs sampling. 

#### Example
```julia
using GibbsSampler
using Distributions

#define prior and proposal distributions
priors = [Normal(2.0,3.0), Normal(3.0,3.0)]

#log of joint probability
function logJoint(params)	
	logPrior= sum(logpdf.(priors, params))
	return logPrior
end
alg = [MH()]
sample_alg =Dict(
	1 => [1, Normal(2.0,3.0)],
	2 => [1, Normal(3.0,3.0)]
)
# Sample from the posterior using Gibbs sampler.
chn = GibbsSampler.gibbs(alg, sample_alg, logJoint;itr = 10000, chain_type = :mcmcchain)
```
### Use of AdvancedHMC as MCMC sampler
The `adHMC()` struct defined with [GibbsSampler.jl](https://github.com/efmanu/GibbsSampler.jl) package is used to select MCMC sampler for each parameter in Gibbs sampling. 

```julia
#select MCMC sampler as vector with adHMC() struct with same length of proposal distribution
alg = [adHMC()]
sample_alg =Dict(
	1 => [1],
	2 => [1]
)

# Sample from the posterior using Gibbs sampler.
chn = GibbsSampler.gibbs(alg, sample_alg, logJoint;itr = 10000, chain_type = :mcmcchain)
```

### Use of AdvancedHMC as MCMC sampler
The `adNUTS()` struct defined with [GibbsSampler.jl](https://github.com/efmanu/GibbsSampler.jl) package is used to select MCMC sampler for each parameter in Gibbs sampling. 

```julia
#select MCMC sampler as vector with adNUTS() struct with same length of proposal distribution
alg = [adNUTS()]
sample_alg =Dict(
	1 => [1],
	2 => [1]
)

# Sample from the posterior using Gibbs sampler.
chn = GibbsSampler.gibbs(alg, sample_alg, logJoint;itr = 10000, chain_type = :mcmcchain)
```

### Use of different MCMC sampler for each parameter

```julia
#select MCMC sampler as vector the same length of proposal distribution
alg = [adNUTS(), MH()]
sample_alg =Dict(
	1 => [1],
	2 => [2, Normal(0.0,1.0)]
)

# Sample from the posterior using Gibbs sampler.
chn = GibbsSampler.gibbs(alg, sample_alg, logJoint;itr = 10000, chain_type = :mcmcchain)
```
