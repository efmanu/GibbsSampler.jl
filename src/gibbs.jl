function gibbs(proposal, logJoint::Function; itr = 100, burn_in = Int(round(itr*0.2))
)	
	states = Dict()
	# prev_params = rand.(proposal)
	param_val = copy(rand.(proposal))
	logJoint_prevs = logJoint(param_val) #logJoint(prev_params)
	logJoint_curs = 0.0
	for i in 1:itr
		states["itr_$i"] =  copy(param_val)#copy(Array(prev_params))
		# param_val = copy(prev_params)
		for (idx,val) in enumerate(proposal)	
			prev_val = param_val[idx]	
			param_val[idx] = copy(rand(val))
			logJoint_curs = logJoint(param_val)
			logα = logJoint_curs - logJoint_prevs
			if (-Random.randexp() < logα)
				states["itr_$i"][idx] = copy(param_val[idx])	
				# prev_params[idx] = copy(param_val[idx])
				logJoint_prevs = logJoint_curs
			else
				param_val[idx] = prev_val
			end
		end
		
	end

	return MHSampler.data_formatting(states, burn_in, itr)
end

