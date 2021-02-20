# Gibbs Sampler.jl

This package helps to generate posterior samples using [Gibbs sampling algorithm](https://en.wikipedia.org/wiki/Gibbs_sampling) from a specified multivariate probability distribution when direct sampling is difficult.
This Julia package supports MH and HMC based algorithms with different automatic differentiation backends.


## Example

```julia

using GibbsSampler
using Distributions

#proposal distribution
proposal = [Normal(0.0,5.0), Normal(0.0,5.0)]

#prior distribution
priors = proposal

model(z) = z[1] + z[2]
output = 5.0

#log joint probability
function logJoint(params)	
	logPrior= sum(logpdf.(priors, params))
	logLikelihood = logpdf(Normal(model(params)), output)
	return logPrior + logLikelihood
end
#sampling
chn = GibbsSampler.gibbs(proposal, logJoint;itr = 10000)
```