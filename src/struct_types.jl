@with_kw struct MH
	n_samples = 10
end

@with_kw struct adHMC
	n_samples = 10
	n_adapts = 5
	backend = ForwardDiff
end
#@with_kw 