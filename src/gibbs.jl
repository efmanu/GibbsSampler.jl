"""
	gibbs(alg, sample_alg, logJoint::Function;  
		revt = [reverse_transform for _ in 1:find_var_count(sample_alg)],
		itr = 100, burn_in = Int(round(itr*sample_alg[:n_grp]*0.2)),
		chain_type=:default, progress = true
	) where {T <: Distribution}

To generate posterior samples using Gibbs sampling algorithm

# Inputs
- `alg` 			:MCMC algorithms based on structs defined in this package as vector eg: `alg = [MH()]`
- `sample_alg` 		:A dictionary maps `alg` to parameter groups index and it contains proposal distribution
if required by the sampling algorithm. 

Eg: `sample_alg = Dict(
	:n_grp => 1,
	1 => Dict(
		:type => :ind,
		:n_vars => 1,
		1 => Dict(
			:proposal => MvNormal(zeros(2),1.0),
			:n_eles => 2,
			:alg => 1
		)
	)
)`

Here key is the paramter group and first index in the value (Vector) maps to `alg` index. Second index in the `sample_alg`
is the proposal distribution. This not mandatory.

- `logJoint` 		:Log PDF as a function
# Keyword Arguments
- `itr` 			:Number of iterations
- `burn_in` 		:Burn in from samples
- `chain_type`	:Sample chain type. default value is `:default`. Samples chains formated using `MCMCChain.jl`
by choosing `chain_type` as `:mcmcchain`
- `progress` 		:To show the sampling progress. Default value is `true`.
# Output
- `chn `			:Generated samples
"""
function gibbs(alg, sample_alg, logJoint::Function;  
	revt = [reverse_transform for _ in 1:find_var_count(sample_alg)],
	itr = 100, burn_in = Int(round(itr*sample_alg[:n_grp]*0.2)),
	chain_type=:default, progress = true
) where {T <: Distribution}
	if progress
		prog = Progress(itr, dt=0.5,
	             barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
	             barlen=50)
	end
	param_val, param_proposal = generate_ini_paramval(sample_alg)
	states = Dict(
		"itr_$(0)" => deepcopy(param_val)
	)
	for i in 1:itr
		if progress
			ProgressMeter.next!(prog, showvalues = [(:iter,i), (:samples, param_val)])
		end	
		p_loc = 1 #varibale count per iteration
		for gx = 1:sample_alg[:n_grp]
			if  (haskey(sample_alg[gx], :n_sub_grp))
				if sample_alg[gx][:type] == :ind
					p_loc, param_val = grp_wrapper_indsubgrp(states, param_val, param_proposal, sample_alg, gx, revt, p_loc, alg, i, logJoint)
					states["itr_$(gx+(i-1)*sample_alg[:n_grp])"] = deepcopy(param_val)
				elseif sample_alg[gx][:type] == :dep
					p_loc, param_val = grp_wrapper_depsubgrp(states, param_val, param_proposal, sample_alg, gx,	revt, p_loc, alg, i, logJoint)
					states["itr_$(gx+(i-1)*sample_alg[:n_grp])"] = deepcopy(param_val)
				else
					throw("Error: No type found")
				end	
			elseif (haskey(sample_alg[gx], :n_vars))
				if sample_alg[gx][:type] == :ind
					p_loc, param_val = grp_wrapper_indvar(states, param_val, param_proposal, sample_alg, gx, revt, p_loc, alg, i, logJoint)		
					states["itr_$(gx+(i-1)*sample_alg[:n_grp])"] = deepcopy(param_val)							
				elseif sample_alg[gx][:type] == :dep
					p_loc, param_val = grp_wrapper_depvar(states, param_val, param_proposal, sample_alg, gx, revt, p_loc, alg, i, logJoint)
					states["itr_$(gx+(i-1)*sample_alg[:n_grp])"] = deepcopy(param_val)
				else
					throw("Error: No type found")
				end	
			else
				throw("Error: No subgroups or no varibles found")
			end

		end
	end
	if progress
		ProgressMeter.finish!(prog)
	end
	delete!(states, "itr_0")
	return format_chain(states, burn_in, itr*sample_alg[:n_grp], chain_type=chain_type)
end

function proposal_sampling(step_wrapper::Function, initial_θ,
 proposalDist, sample_alg::MH)
	model = AdvancedMH.DensityModel(step_wrapper)
	spl = AdvancedMH.RWMH(proposalDist)
	chain = AdvancedMH.sample(model, spl, sample_alg.n_samples; init_params = initial_θ)
	return chain[end].params
end
function proposal_sampling(step_wrapper::Function, initial_θ,
 proposalDist, sample_alg::adHMC)
	initial_nθ = copy(Array{Float64}(form_single_vec(initial_θ)))
	D = length(initial_nθ)
	ℓπ_grad(θ) = return (step_wrapper(θ), sample_alg.backend.gradient(step_wrapper, θ))	
	metric = DiagEuclideanMetric(D)
	hamiltonian = Hamiltonian(metric, step_wrapper, ℓπ_grad)
	initial_ϵ = find_good_stepsize(hamiltonian, initial_nθ)
	integrator = Leapfrog(initial_ϵ)
	proposal = AdvancedHMC.StaticTrajectory(integrator, 1)
	adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
	samples, stats = AdvancedHMC.sample(hamiltonian, proposal, initial_nθ, sample_alg.n_samples, adaptor, sample_alg.n_adapts, verbose =false)
	return reform_data(samples[end], initial_θ)
 end
function proposal_sampling(step_wrapper::Function, initial_θ,
	proposalDist, sample_alg::adNUTS)
	initial_nθ = copy(Array{Float64}(form_single_vec(initial_θ)))
	D = length(initial_nθ)
	ℓπ_grad(θ) = return (step_wrapper(θ),sample_alg.backend.gradient(step_wrapper, θ))	
	metric = DiagEuclideanMetric(D)
	hamiltonian = Hamiltonian(metric, step_wrapper, ℓπ_grad)
	initial_ϵ = find_good_stepsize(hamiltonian, initial_nθ)
	integrator = Leapfrog(initial_ϵ)
	proposal =  AdvancedHMC.NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
	adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
	samples, stats = AdvancedHMC.sample(hamiltonian, proposal, initial_nθ, sample_alg.n_samples, adaptor, sample_alg.n_adapts, verbose =false)
	return reform_data(samples[end], initial_θ)
end





