"""
	MH
MH is a struct to choose MH MCMC sampling algorithm
# Fields
- `n_samples`	: Number of iterations. Default value is 10

# Example
a = MH(n_samples = 4)
"""
@with_kw struct MH
	n_samples = 2
end

"""
	adHMC
adHMC is a struct to choose HMC MCMC sampling algorithm
# Fields
- `n_samples`	: Number of iterations
- `n_adapts`	: Adaptation
- `backend` 	: Automatic differentiation backend

# Example

a = adHMC(
n_samples = 10,
n_adapts = 5,
backend = ForwardDiff
)
"""
@with_kw struct adHMC
	n_samples = 2
	n_adapts = 5
	backend = ForwardDiff
end


"""
	adNUTS
adNUTS is a struct to choose NUTS MCMC sampling algorithm
# Fields
- `n_samples`	: Number of iterations
- `n_adapts`	: Adaptation
- `backend` 	: Automatic differentiation backend

# Example

a = adNUTS(
n_samples = 10,
n_adapts = 5,
backend = ForwardDiff
)
"""
@with_kw struct adNUTS
	n_samples = 2
	n_adapts = 5
	backend = ForwardDiff
end