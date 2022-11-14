function sparsity(context)
    model = context.models[1]
    work = context.work
    params = work.params
    results = context.results.model_results[1]
    trends = results.trends

    compute_steady_state!(context, Dict{String, Any}())

    steadystate = trends.endogenous_steady_state
    endogenous = repeat(steadystate, 3)
    exogenous = trends.exogenous_steady_state
    ws = DynamicWs(context)
    jacobian = get_dynamic_jacobian!(
        ws,
        params,
        endogenous,
        exogenous,
        steadystate,
        model,
        2,
    )
    n = length(steadystate)
    println("Sparsity")
    @views begin
        println("A0: $(nnz(jacobian[:, 1:n])/n^2)")
        println("A1: $(nnz(jacobian[:, n .+ (1:n)])/n^2)")
        println("A2: $(nnz(jacobian[:, 2*n .+ (1:n)])/n^2)")
    end
end
