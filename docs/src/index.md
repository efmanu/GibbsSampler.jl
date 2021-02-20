# GibbsSampler.jl

```@meta
CurrentModule = GibbsSampler
DocTestSetup = quote
    using GibbsSampler
end
```

This package helps to generate posterior samples using [Gibbs sampling algorithm](https://en.wikipedia.org/wiki/Gibbs_sampling) from a specified multivariate probability distribution when direct sampling is difficult.
This Julia package supports MH and HMC based algorithms with different automatic differentiation backends.

## Gibbs Sampling
```@docs
gibbs(proposal::Vector{T}, logJoint::Function; 
		sample_alg = [MH() for _ in 1:length(proposal)], 
		itr = 100, burn_in = Int(round(itr*0.2))
	) where {T <: Distribution}
```