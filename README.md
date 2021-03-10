# GibbsSampler.jl

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://efmanu.github.io/GibbsSampler.jl/dev/
[![][docs-dev-img]][docs-dev-url]

This package helps to generate posterior samples using [Gibbs sampling algorithm](https://en.wikipedia.org/wiki/Gibbs_sampling) from a specified multivariate probability distribution when direct sampling is difficult. This Julia package supports MH, HMC and NUTS based algorithms with different automatic differentiation backends.


## Example

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

param_names = ["α", "β", "γ", "δ"]

#define logjoint function
logJoint(params) = sum(logpdf.(prior, params))

#sample
chn = gibbs(alg, sample_alg, logJoint, itr = 10000, chain_type = :mcmcchain, param_names = param_names)
```