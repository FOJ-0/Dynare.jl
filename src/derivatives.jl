# I. Derivatives of the steady state w.r parameters
struct DerivativesWs
    dx_dp::Matrix{Float64}
    function DerivativesWs(n::Int, m::Int)
        LU_df_dx::SuiteSparse.UMFPACK.UmfpackLU{Float64, Int64}
        new(dense_df_dx, df_dx_inv, result)
    end
end

function SSDerivatives!(dx_dp, df_dx, dfstatic_dp)
    # Perform Sparse LU decomposition
    LU_df_dx = lu(df_dx)

    # Get dx_dp
#    copyto!(dx_dp, -(LU_df_dx \ df_dp))
    ldiv!(dx_dp, LU_df_dx, dfstatic_dp)
    lmul!(-1.0, dx_dp)

    return dx_dp
end

function getABC(jacobian, model)
    n = model.endogenous_nbr
    @views A = jacobian[:, 2*n .+ (1:n)]
    @views B = jacobian[:, n .+ (1:n)]
    @views C = jacobian[: , 1:n]
    return A, B, C
end

function dynamic_derivatives_wr_state_and_parameters(context)
    model = context.models[1]
    
    n = model.endogenous_nbr
    m = model.parameter_nbr
    steadystate = context.results.model_results[1].trends.endogenous_steady_state
    endogenous3 = repeat(steadystate, 3)
    lli = model.lead_lag_incidence
    endogenous = endogenous3[findall(!iszero, vec(lli'))]
    (dr_dp, gp) = Dynare.DFunctions.SparseDynamicParametersDerivatives!(endogenous, exogenous, params, steadystate, 2, [], [])
    
    dA_dp = zeros(n, n, m)
    dB_dp = zeros(n, n, m)
    dC_dp = zeros(n, n, m)
    
    # derivatives of A, B, C w.r. to parameter
    @views begin
        k1 = findall(!iszero, lli[3, :])
        k2 = lli[3, k1]
        dA_dp[:, k1, :] .= gp[:, k2, :]
        k1 = findall(!iszero, lli[2, :])
        k2 = lli[2, k1]
        dB_dp[:, k1, :] .= gp[:, k2, :]
        k1 = findall(!iszero, lli[1, :])
        k2 = lli[1, k1]
        dC_dp[:, k1, :] .= gp[:, k2, :]
    end
    return dA_dp, dB_dp, dC_dp
end

function ABCderivatives(jacobian, context, wsd)
    model = context.models[1]
    params = context.work.params
    trends = context.results.model_results[1].trends
    steadystate = trends.endogenous_steady_state
    endogenous3 = repeat(steadystate, 3)
    
    dss_dx = Dynare.get_static_jacobian!(wss, params, steadystate, exogenous, model) 
    SSDerivatives!(dss_dp, dss_dx, dfstatic_dp)
    values = zeros(size(model.dynamic_g2_sparse_indices, 1))
    Dynare.get_dynamic_derivatives2!(
        wsd,
        params,
        endogenous3,
        exogenous,
        steadystate,
        values,
        model,
    )
    dA_dp, dB_dp, dC_dp = dynamic_derivatives_wr_state_and_parameters(context)
    F2 = Matrix(wsd.derivatives[2])
    for p in 1:7
        k1 = collect(1:15)
        k2 = collect(16:30)
        k3 = collect(31:45)
        for i=1:3
            for j = 1:15
                dA_dp[:, :, p] .+= dss_dp[j, p] .* F2[:, k3]
                k3 .+= 47
            end
            for j = 1:15
                dB_dp[:, :, p] .+= dss_dp[j, p] .* F2[:, k2]
                k2 .+= 47
            end
            for j = 1:15
                dC_dp[:, :, p] .+= dss_dp[j, p] .* F2[:, k1]
                k1 .+= 47
            end
        end
    end
    return dA_dp, dB_dp, dC_dp
end

function d1_lre_solution(dA_dp, dB_dp, dC_dp, g1, A, B, C, state_indices, endo_nbr, param_nbr)
    # solution to UQME
    X = zeros(endo_nbr, endo_nbr)
    # set nonzero columns
    X[:, state_indices] .= g1

    #Generalized Sylvester: ax + bxc = d
    # a = A*X + B
    a = copy(B)
    mul!(a, A, X, 1, 1)
    b = Matrix(A)
    c = X
    X2 = X*X
    dX = [zeros(endo_nbr, endo_nbr) for i in 1:param_nbr]
    
    
    # order=1
    # GSws = GeneralizedSylvesterWs(n,n,n,order)
    # Solve UQME using generalized_sylvester_solver!
    LU = lu((kron(I(endo_nbr), a) + kron(c', b)))
    XX = zeros(endo_nbr, endo_nbr)
    for i in 1:param_nbr
        fill!(XX, 0.0)  
        @views begin
            mul!(XX, dA_dp[:,:,i], X2)
            mul!(XX, dB_dp[:,:,i], X, true, true) 
            XX .+= dC_dp[:,:,i]
        end
        #    generalized_sylvester_solver!(a, b, c, d[:,:,i], order, GSws)
        d = copy(XX)
        ldiv!(LU, vec(XX))
        @views dX[i] .= .-XX
    end
    return dX
end


