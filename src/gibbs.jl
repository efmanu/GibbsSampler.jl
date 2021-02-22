"""
	function gibbs(proposal::Vector{T}, logJoint::Function; 
		sample_alg = [MH() for _ in 1:length(proposal)], 
		itr = 100, burn_in = Int(round(itr*0.2))
	) where {T <: Distribution}

To generate posterior samples using Gibbs sampling algorithm

# Inputs
- proposal 		:proposal distributions as vector
- logJoint 		:Log PDF as a function
# Keyword Arguments
- sample_alg 	:MCMC sampling for each parameter
- itr 			:Number of iterations
- burn_in 		:Burn in from samples
# Output
- chn 			:Generated samples
"""
function gibbs(proposal::Vector{T}, logJoint::Function; 
	sample_alg = [MH() for _ in 1:length(proposal)], 
	itr = 100, burn_in = Int(round(itr*0.2))
) where {T <: Distribution}
	states = Dict()
	param_val = copy(rand.(proposal))
	for i in 1:itr
		states["itr_$i"] =  copy(param_val)
		for (idx,val) in enumerate(proposal)
			function step_wrapper(new_param)
				nw_param_val = [param_val[1:idx-1]..., reverse_transform(new_param), param_val[idx+1:end]...]
				return logJoint(nw_param_val)
			end	
			global g_wrapper = step_wrapper
			if i == 1
				initial_θ = rand(val)
			else
				initial_θ = states["itr_$(i-1)"][idx]
			end			
			param_val[idx] = reverse_transform(proposal_sampling(step_wrapper, initial_θ, val, sample_alg[idx]))

			states["itr_$i"][idx] = param_val[idx]
		end		
	end
	return states
end

function proposal_sampling(step_wrapper::Function, initial_θ,
	proposalDist::Distribution, sample_alg::adHMC)
	D = length(initial_θ)
	ℓπ_grad(θ) = return (step_wrapper(θ), sample_alg.backend.gradient(step_wrapper, θ))	
	metric = DiagEuclideanMetric(D)
	hamiltonian = Hamiltonian(metric, step_wrapper, ℓπ_grad)
	initial_ϵ = find_good_stepsize(hamiltonian, forward_transform(initial_θ))
	integrator = Leapfrog(initial_ϵ)
	proposal = AdvancedHMC.StaticTrajectory(integrator, 1)
	adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
	samples, stats = AdvancedHMC.sample(hamiltonian, proposal, forward_transform(initial_θ), sample_alg.n_samples, adaptor, sample_alg.n_adapts, verbose =false)
	return samples[end]
end

function proposal_sampling(step_wrapper::Function, initial_θ,
	proposalDist::Distribution, sample_alg::adNUTS)
	D = length(initial_θ)
	ℓπ_grad(θ) = return (step_wrapper(θ), sample_alg.backend.gradient(step_wrapper, θ))	
	metric = DiagEuclideanMetric(D)
	hamiltonian = Hamiltonian(metric, step_wrapper, ℓπ_grad)
	initial_ϵ = find_good_stepsize(hamiltonian, forward_transform(initial_θ))
	integrator = Leapfrog(initial_ϵ)
	proposal =  AdvancedHMC.NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
	adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
	samples, stats = AdvancedHMC.sample(hamiltonian, proposal, forward_transform(initial_θ), sample_alg.n_samples, adaptor, sample_alg.n_adapts, verbose =false)
	return samples[end]
end

function proposal_sampling(step_wrapper::Function, initial_θ,
 proposalDist::Distribution, sample_alg::MH)
	model = AdvancedMH.DensityModel(step_wrapper)
	spl = AdvancedMH.RWMH(proposalDist)
	chain = AdvancedMH.sample(model, spl, sample_alg.n_samples; init_params = initial_θ)
	return chain[end].params
end





