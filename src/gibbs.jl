function gibbs(proposal, logJoint::Function; itr = 100, burn_in = Int(round(itr*0.2))
)	
	states = Dict()
	prev_params = rand.(proposal)
	logJoint_prevs = logJoint(prev_params)
	logJoint_curs = 0.0
	for i in 1:itr
		acc = falses(length(prev_params))
		param_val = copy(prev_params)
		for (idx,val) in enumerate(proposal)			
			param_val[idx] = rand(val)
			logJoint_curs = logJoint(param_val)
			logα = logJoint_curs - logJoint_prevs
			if (-Random.randexp() < logα)	
				acc[idx] = true
			end
		end
		if(all(acc))
			prev_params = copy(param_val)
			logJoint_prevs = logJoint_curs
		end
		states["itr_$i"] = Array(prev_params)
	end

	return MHSampler.data_formatting(states, burn_in, itr)
end

