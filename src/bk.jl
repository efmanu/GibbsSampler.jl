for i in 1:itr
	for idx in 1:sample_alg[:grp]
		if sample_alg[idx][:type] == :ind #independent samples, sample indepedently
			#create a thinning wrapper for each params
				#create proposal distribution
				#identify initial condition
				#call sampler function
			#form group sample value
			#update sampler value for all groups together
		else #dependent samples, so sample together
			#create a thinning wrapper 
			#create proposal distribution
			#identify initial condition
			#call sampler function
			#update sampler value for all groups together
		end
	end
end

#mh sampler function
#proposal distribution
#initial condition
#hmc sampler function
