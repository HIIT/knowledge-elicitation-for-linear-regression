function [fa, si, converged] = linreg_sns_ep(y, x, pr, op, feedbacks)
% -- Likelihood:
%    p(y_i|x_i,w,sigma2) = N(y_i|w'x_i, sigma2)
%    p(f_j|w_j,eta2) = N(f_j|w_j, eta2)
% -- Prior:
% p(w_j|gamma_j=1) = Normal(w_j|0, tau2)
% p(w_j|gamma_j=0) = delta(w_j)
% p(gamma_j|rho) = Bernoulli(gamma_j|rho)
% -- Approximation;
% q(w) = Normal(w|Mean_w, Var_w), Var_w = Tau_w^-1
% q(gamma) = \prod Bernoulli(\gamma_j|p_gamma_j)
%
% Inputs:
% y          target values (n x 1)
% x          covariates (n x m)
% pr         prior parameters (struct)
% op         options (struct)
% feedbacks  values (1st column) and indices (2nd column) of feedback (n_feedbacks x 2)
%
% Outputs:
% fa         EP posterior approximation (struct)
% si         EP site terms (struct)
% converged  did EP converge or hit max_iter (1/0)
%
% Tomi Peltola, tomi.peltola@aalto.fi

if nargin < 5
    feedbacks = [];
end

[n, m] = size(x);
pr.n = n;
pr.m = m;
n_feedbacks = size(feedbacks, 1);

%% initialize
si.prior.w.mu = zeros(m, 1);
si.prior.w.tau = (1 / pr.tau2) * ones(m, 1);
si.prior.gamma.a = ones(m, 1);
si.prior.gamma.b = ones(m, 1);
S_f = zeros(m, m);
F_f = zeros(m, 1);
if n_feedbacks > 0
    for i = 1:n_feedbacks
        S_f(feedbacks(i, 2), feedbacks(i, 2)) = 1;
        F_f(feedbacks(i, 2)) = feedbacks(i, 1);
    end
end
si.lik.w.Tau = (1 / pr.sigma2) * (x' * x) + (1 / pr.eta2) * S_f;
si.lik.w.Mu = (1 / pr.sigma2) * x' * y + (1 / pr.eta2) * F_f;

% full approximation
fa = compute_full_approximation(si, pr);

% convergence diagnostics
conv.P_gamma_old = Inf * ones(m, 1);
conv.z_old = Inf * ones(m, 1);

%% loop parallel EP
for iter = 1:op.max_iter
    %% prior updates
    % cavity
    ca_prior = compute_prior_cavity(fa, si.prior, pr);
    
    % moments of tilted dists
    [ti_prior, z] = compute_prior_tilt(ca_prior, pr);
    
    % site updates
    si.prior = site_updates_prior(si.prior, ca_prior, ti_prior, op);
    
    %% full approx update
    fa = compute_full_approximation(si, pr);
    
    %% show progress and check for convergence
    [converged, conv] = report_progress_and_check_convergence(conv, iter, z, fa, op);
    if converged
        break
    end
    
    %% update damp
    op.damp = op.damp * op.damp_decay;
end

end


function ca = compute_prior_cavity(fa, si, pr)

m = pr.m;

tmp = fa.w.Tau_chol \ eye(m);
var_w = sum(tmp.^2)';

denom = (1 - si.w.tau .* var_w);
ca.w.tau = denom ./ var_w;
ca.w.mean = (fa.w.Mean - var_w .* si.w.mu) ./ denom;

ca.gamma.a = pr.rho * ones(m, 1);
ca.gamma.b = (1 - pr.rho) * ones(m, 1);

end


function [ti, z] = compute_prior_tilt(ca, pr)

t = ca.w.tau + 1 ./ pr.tau2;

g_var = 1 ./ ca.w.tau; % for gamma0
mcav2 = ca.w.mean.^2;
log_z_gamma0 = log(ca.gamma.b) - 0.5 * log(g_var) - 0.5 * mcav2 ./ g_var;
g_var = pr.tau2 + g_var; % for gamma1
log_z_gamma1 = log(ca.gamma.a) - 0.5 * log(g_var) - 0.5 * mcav2 ./ g_var;
z_gamma0 = exp(log_z_gamma0 - log_z_gamma1);
z_gamma1 = ones(size(log_z_gamma1));
z = 1 + z_gamma0;

ti.w.mean = z_gamma1 .* (ca.w.tau .* ca.w.mean) ./ t ./ z;
e2_w_tilt = z_gamma1 .* (1 ./ t + 1 ./ t.^2 .* (ca.w.tau .* ca.w.mean).^2) ./ z;
ti.w.var = e2_w_tilt - ti.w.mean.^2;

ti.gamma.mean = z_gamma1 ./ z;

end


function [si, nonpositive_cavity_vars, nonpositive_site_var_proposals] = site_updates_prior(si, ca, ti, op)

nonpositive_site_var_proposals = false;

% skip negative cavs
update_inds = ca.w.tau(:) > 0;
nonpositive_cavity_vars = ~all(update_inds);

new_tau_w_site = 1 ./ ti.w.var - ca.w.tau;

switch op.robust_updates
    case 0
    case 1
        inds_tmp = new_tau_w_site(:) > 0;
        nonpositive_site_var_proposals = ~all(inds_tmp);
        update_inds = update_inds & inds_tmp;
    case 2
        inds = new_tau_w_site(:) <= 0;
        new_tau_w_site(inds) = op.min_site_prec;
        ti.w.var(inds) = 1./(op.min_site_prec + ca.w.tau(inds));
end
new_mu_w_site = ti.w.mean ./ ti.w.var - ca.w.tau .* ca.w.mean;
si.w.tau(update_inds) = (1 - op.damp) * si.w.tau(update_inds) + op.damp * new_tau_w_site(update_inds);
si.w.mu(update_inds) = (1 - op.damp) * si.w.mu(update_inds) + op.damp * new_mu_w_site(update_inds);

% TODO: use log scale for a/b_gamma computations?
si.gamma.a(update_inds) = exp((1 - op.damp) * log(si.gamma.a(update_inds)) + op.damp * log(ti.gamma.mean(update_inds) ./ ca.gamma.a(update_inds)));
si.gamma.b(update_inds) = exp((1 - op.damp) * log(si.gamma.b(update_inds)) + op.damp * log((1 - ti.gamma.mean(update_inds)) ./ ca.gamma.b(update_inds)));

end


function fa = compute_full_approximation(si, pr)

% m x m and m x 1
fa.w.Tau = si.lik.w.Tau + diag(si.prior.w.tau);
fa.w.Tau_chol = chol(fa.w.Tau, 'lower');
fa.w.Mu = si.lik.w.Mu + si.prior.w.mu;
fa.w.Mean = fa.w.Tau_chol' \ (fa.w.Tau_chol \ fa.w.Mu);

fa.P_gamma = si.prior.gamma.a .* pr.rho;

end


function [converged, conv] = report_progress_and_check_convergence(conv, iter, z, fa, op)

conv_z = mean(abs(z(:) - conv.z_old(:)));
conv_P_gamma = mean(abs(fa.P_gamma(:) - conv.P_gamma_old(:)));

% % PED: I commented the next few lines for now
% if op.verbosity > 0 && mod(iter, op.verbosity) == 0
%     fprintf(1, '%d, conv = [%.2e %.2e], damp = %.2e\n', iter, conv_z, conv_P_gamma, op.damp);
% end

%converged = conv_z < op.threshold && conv_P_gamma < op.threshold;
converged = conv_P_gamma < op.threshold;

conv.z_old = z;
conv.P_gamma_old = fa.P_gamma;

end