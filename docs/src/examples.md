# Examples
This section contains different examples that describes the usage of [GibbsSampler.jl](https://github.com/efmanu/GibbsSampler.jl) package.

## Different MCMC Samplers for parameter sampling

The [GibbsSampler.jl](https://github.com/efmanu/GibbsSampler.jl) allows the use of **MH**, **HMC**, **NUTS** MCMC samplers with the help of [AdvancedMH.jl](https://github.com/TuringLang/AdvancedMH.jl) and [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl) package. The `GibbsSampler.gibbs(...)` function has a keyword argument named `sample_alg`, which decides the MCMC sampler and intialized as vector with `MH()` struct with same length as that of proposal distribution vector.

### Use of AdvancedMH as MCMC sampler
The `MH()` struct defined with [GibbsSampler.jl](https://github.com/efmanu/GibbsSampler.jl) package is used to select MCMC sampler for each parameter in Gibbs sampling. 

#### Example
```julia
using GibbsSampler
using Distributions

#define prior and proposal distributions
proposal = [Normal(2.0,3.0), Normal(3.0,3.0)]
priors = proposal

#select MCMC sampler as vector with MH() struct with same length of proposal distribution
sample_alg = [MH() for _ in 1:length(proposal)]


#log of joint probability
function logJoint(params)	
	logPrior= sum(logpdf.(priors, params))
	return logPrior
end
# Sample from the posterior using Gibbs sampler.
chn = GibbsSampler.gibbs(proposal, logJoint;itr = 10000, sample_alg=sample_alg, chain_type = :mcmcchain)
```
### Use of AdvancedHMC as MCMC sampler
The `adHMC()` struct defined with [GibbsSampler.jl](https://github.com/efmanu/GibbsSampler.jl) package is used to select MCMC sampler for each parameter in Gibbs sampling. 

```julia
#select MCMC sampler as vector with adHMC() struct with same length of proposal distribution
sample_alg = [adHMC() for _ in 1:length(proposal)]

# Sample from the posterior using Gibbs sampler.
chn = GibbsSampler.gibbs(proposal, logJoint;itr = 10000, sample_alg=sample_alg)
```

### Use of AdvancedHMC as MCMC sampler
The `adNUTS()` struct defined with [GibbsSampler.jl](https://github.com/efmanu/GibbsSampler.jl) package is used to select MCMC sampler for each parameter in Gibbs sampling. 

```julia
#select MCMC sampler as vector with adNUTS() struct with same length of proposal distribution
sample_alg = [adNUTS() for _ in 1:length(proposal)]

# Sample from the posterior using Gibbs sampler.
chn = GibbsSampler.gibbs(proposal, logJoint;itr = 10000, sample_alg=sample_alg, chain_type = :mcmcchain)
```

### Use of different MCMC sampler for each parameter

```julia
#select MCMC sampler as vector the same length of proposal distribution
sample_alg = [adNUTS(), MH()]

# Sample from the posterior using Gibbs sampler.
chn = GibbsSampler.gibbs(proposal, logJoint;itr = 10000, sample_alg=sample_alg, chain_type = :mcmcchain)
```
