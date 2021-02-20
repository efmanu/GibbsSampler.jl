function gibbs(proposal::Vector{T}, logJoint::Function; 
	sample_alg = [MH() for _ in 1:length(proposal)], 
	itr = 100, burn_in = Int(round(itr*0.2))
) where {T <: Distribution}
	states = Dict()
	param_val = copy(rand.(proposal))
	logJoint_prevs = logJoint(param_val)
	logJoint_curs = 0.0
	for i in 1:itr
		states["itr_$i"] =  copy(param_val)
		for (idx,val) in enumerate(proposal)
			function step_wrapper(new_param) 
				nw_param_val = vcat(param_val[1:idx-1], new_param[1], param_val[idx+1:end])
				return logJoint(nw_param_val)
			end	
			param_val[idx] = proposal_sampling(step_wrapper, [rand(val)], val, sample_alg[idx])
			states["itr_$i"][idx] = param_val[idx]
		end		
	end
	return MHSampler.data_formatting(states, burn_in, itr)
end

function proposal_sampling(step_wrapper::Function, initial_θ,
	proposalDist::Distribution, sample_alg::adHMC; D =1)
	metric = DiagEuclideanMetric(D)
	hamiltonian = Hamiltonian(metric, step_wrapper, sample_alg.backend)
	initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
	integrator = Leapfrog(initial_ϵ)
	proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
	adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
	samples, stats = AdvancedHMC.sample(hamiltonian, proposal, initial_θ, sample_alg.n_samples, adaptor, sample_alg.n_adapts, verbose =false)
	return samples[end][1]
end

function proposal_sampling(step_wrapper::Function, initial_θ,
 proposalDist::Distribution, sample_alg::MH; D =1)
	model = AdvancedMH.DensityModel(step_wrapper)
	spl = AdvancedMH.RWMH(proposalDist)
	chain = AdvancedMH.sample(model, spl, sample_alg.n_samples)
	return chain[end].params[1]
end




