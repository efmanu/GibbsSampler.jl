Base.vec(x::Float64) = [x]
function forward_transform(x)
	return vec(x)
end

function reverse_transform(x)
	if (length(x) == 1) && (x isa Vector)
		return x[1]
	else
		return x
	end
end

function form_vector(param_val, idx, new_param)
	nw_param_val = [param_val[1:idx-1]..., new_param, param_val[idx+1:end]...]
	return nw_param_val
end

function format_chain(states, burn_in, itr; chain_type=:default)
	chain =DataFrame();
	if(!isempty(states))
		lps = length(states["itr_1"])
		param_names =[]
		for ln in 1:lps
			param_st = forward_transform(states["itr_1"][ln])
			for ps in 1:length(param_st)
				push!(param_names,"param[$(ln)][$(ps)]")
			end			
		end
		chain.var = param_names
		for i in (burn_in+1):itr
			all_val =[]
			for ln in 1:lps	
				param_st = forward_transform(states["itr_$(i)"][ln])
				push!(all_val, param_st...)
			end
			chain[!,Symbol((i-burn_in))] = all_val
		end
		if(chain_type == :default)
			return chain
		elseif(chain_type == :mcmcchain)
			return Chains(Matrix{Float64}(Array(chain[:,2:end])'), param_names)
		else
			throw("Error: Chain type not found")
		end
	end
	return chain
end