"""
	gibbs(alg, sample_alg, logJoint::Function;  
		revt = [reverse_transform for _ in 1:length(sample_alg)],
		itr = 100, burn_in = Int(round(itr*0.2)),
		chain_type=:default, progress = true
	) where {T <: Distribution}

To generate posterior samples using Gibbs sampling algorithm

# Inputs
- alg 			:MCMC algorithms based on structs defined in this package as vector eg: alg = [MH()]
- sample_alg 	:A dictionary maps `alg` to parameter groups index and it contains proposal distribution
if required by the sampling algorithm. 
Eg: sample_alg =Dict(
	1 => [1, Normal(2.0,3.0)],
	2 => [1, Normal(3.0,3.0)]
)
Here key is the paramter group and first index in the value (Vector) maps to `alg` index. Second index in the `sample_alg`
is the proposal distribution. This not mandatory.
- logJoint 		:Log PDF as a function
# Keyword Arguments
- itr 			:Number of iterations
- burn_in 		:Burn in from samples
- chain_type	:Sample chain type. default value is `:default`. Samples chains formated using `MCMCChain.jl`
by choosing `chain_type` as `:mcmcchain`
- progress 		:To show the sampling progress. Default value is `true`.
# Output
- chn 			:Generated samples
"""
function gibbs(alg, sample_alg, logJoint::Function;  
	revt = [reverse_transform for _ in 1:length(sample_alg)],
	itr = 100, burn_in = Int(round(itr*0.2)),
	chain_type=:default, progress = true
) where {T <: Distribution}
	states = Dict()
	lens = length(sample_alg)
	param_val = copy(rand(lens))
	if progress
		prog = Progress(itr, dt=0.5,
	             barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
	             barlen=50)
	end
	val = check_sample_alg(alg, sample_alg)
	for i in 1:itr
		if progress
			ProgressMeter.next!(prog, showvalues = [(:iter,i), (:samples, param_val)])
		end
		states["itr_$i"] =  copy(param_val)
		for idx in 1:lens
			function step_wrapper(new_param)
				nw_param_val = [param_val[1:idx-1]..., revt[idx](new_param), param_val[idx+1:end]...]
				return logJoint(nw_param_val)
			end	

			if i == 1
				initial_θ = rand(val[idx])
			else
				initial_θ = states["itr_$(i-1)"][idx]
			end			
			param_val[idx] = revt[idx](proposal_sampling(step_wrapper, initial_θ, val[idx], alg[sample_alg[idx][1]]))

			states["itr_$i"][idx] = param_val[idx]
		end		
	end
	if progress
		ProgressMeter.finish!(prog)
	end
	return format_chain(states, burn_in, itr, chain_type=chain_type)
end

function proposal_sampling(step_wrapper::Function, initial_θ,
	proposalDist, sample_alg::adHMC)
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
	proposalDist, sample_alg::adNUTS)
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





