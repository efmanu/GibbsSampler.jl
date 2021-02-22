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

function reshape_chain(states, itr)
end