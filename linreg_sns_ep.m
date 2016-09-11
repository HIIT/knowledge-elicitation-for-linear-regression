function [fa, si, converged, subfunctions] = linreg_sns_ep(y, x, pr, op, w_feedbacks, gamma_feedbacks, si)
% -- Likelihood (y are data, f are feedbacks):
%    p(y_i|x_i,w,sigma2) = N(y_i|w'x_i, sigma2)
%    p(f_w_j|w_j,eta2) = N(f_w_j|w_j, eta2)
%    p(f_gamma_j|gamma_j) = I(gamma_j=1) Bernoulli(f_gamma_j|p_u) + I(gamma_j=0) Bernoulli(f_gamma_j|1-p_u)
% -- Prior:
%    p(w_j|gamma_j=1) = Normal(w_j|0, tau2)
%    p(w_j|gamma_j=0) = delta(w_j)
%    p(gamma_j|rho) = Bernoulli(gamma_j|rho)
%    p(rho) = Beta(rho|rho_a, rho_b)
%    p(sigma2^-1) = Gamma(sigma2^-1|sigma2_a,sigma2_b) or fixed sigma2
% -- Approximation;
%    q(w) = Normal(w|w.Mean, w_Var), w_Var = w.Tau^-1
%    q(gamma) = \prod_j Bernoulli(gamma_j|gamma.p_j)
%    q(sigma2^-1) = Gamma(sigma2^-1|sigma2_a,sigma2_b), mean: sigma2.imean
%    q(rho) = Beta(rho|rho.a,rho.b)
%
%    sigma2 and rho are updated using VB (if not fixed), other terms using EP.
%
% Inputs:
% y                target values (n x 1)
% x                covariates (n x m)
% pr               prior and other fixed model parameters (struct)
% op               options for the EP algorithm (struct)
% w_feedbacks      values (1st column) and indices (2nd column) of feedback (n_w_feedbacks x 2)
% gamma_feedbacks  values (1st column, 0/1) and indices (2nd column) of feedback (n_gamma_feedbacks x 2)
% si               if given, (some of) site parameters initialized to these
%
% Outputs:
% fa         EP posterior approximation (struct)
% si         EP site terms (struct)
% converged  did EP converge or hit max_iter (1/0)
%
% Tomi Peltola, tomi.peltola@aalto.fi

if nargin < 5
    w_feedbacks = [];
end

if nargin < 6
    gamma_feedbacks = [];
end

[n, m] = size(x);
pr.n = n;
pr.m = m;
pr.p_u_nat = log(pr.p_u) - log1p(-pr.p_u);
pr.yy = y' * y; % precompute
pr.xy = x' * y; % precompute
pr.xx = x' * x; % precompute
n_w_feedbacks = size(w_feedbacks, 1);
n_gamma_feedbacks = size(gamma_feedbacks, 1);

%% initialize (if si is given, prior sites are not re-initialized, but likelihood is)
if nargin < 7 || isempty(si)
    si.w_prior.normal_mu = zeros(m, 1);
    si.w_prior.normal_tau = (1 / pr.tau2) * ones(m, 1);
    si.w_prior.bernoulli_p_nat = zeros(m, 1);
end
S_f = zeros(m, m);
F_f = zeros(m, 1);
if n_w_feedbacks > 0
    for i = 1:n_w_feedbacks
        S_f(w_feedbacks(i, 2), w_feedbacks(i, 2)) = 1;
        F_f(w_feedbacks(i, 2)) = w_feedbacks(i, 1);
    end
end
si.w_feedback.normal_Tau = (1 / pr.eta2) * S_f;
si.w_feedback.normal_Mu = (1 / pr.eta2) * F_f;
if isfield(pr, 'sigma2_prior') && pr.sigma2_prior
    si.y_lik.gamma_a = 0.5 * n;
    si.y_lik.gamma_b = 0.5 * pr.yy;
    sigma2_imean = (pr.sigma2_a + si.y_lik.gamma_a) / (pr.sigma2_b + si.y_lik.gamma_b);
    si.y_lik.normal_Tau = sigma2_imean * pr.xx;
    si.y_lik.normal_Mu = sigma2_imean * pr.xy;
else
    si.y_lik.normal_Tau = (1 / pr.sigma2) * pr.xx;
    si.y_lik.normal_Mu = (1 / pr.sigma2) * pr.xy;
    pr.sigma2_prior = 0;
end
si.gamma_feedback.bernoulli_p_nat = zeros(m, 1);

if isfield(pr, 'rho_prior') && pr.rho_prior
    rho_ = pr.rho_a / (pr.rho_a + pr.rho_b);
    si.gamma_prior.bernoulli_p_nat = log(rho_) - log1p(-rho_);
    si.gamma_prior.beta_a = zeros(m, 1);
    si.gamma_prior.beta_b = zeros(m, 1);
else
    si.gamma_prior.bernoulli_p_nat = log(pr.rho) - log1p(-pr.rho);
    pr.rho_prior = 0;
end

% full approximation
fa = compute_full_approximation(si, pr);

% convergence diagnostics
conv.P_gamma_old = Inf * ones(m, 1);
conv.z_old = Inf * ones(m, 1);

%% loop parallel EP
for iter = 1:op.max_iter
    %% w prior updates
    % cavity
    ca_w_prior = compute_sns_prior_cavity(fa, si.w_prior, pr);
    
    % moments of tilted dists
    [ti_w_prior, z_w] = compute_sns_prior_tilt(ca_w_prior, pr);
    
    % site updates
    si.w_prior = update_sns_prior_sites(si.w_prior, ca_w_prior, ti_w_prior, op);
    
    % full approx update
    fa = compute_full_approximation_w(fa, si, pr);
    fa = compute_full_approximation_gamma(fa, si, pr);

    %% gamma prior updates, EP for gamma, VB for rho
    if pr.rho_prior
        % VB
        si.gamma_prior = update_bernoulli_sites_vb(si.gamma_prior, fa.gamma.p, op);
        
        fa = compute_full_approximation_rho(fa, si, pr);
        
        % EP
        si.gamma_prior = update_bernoulli_sites_ep(si.gamma_prior, fa.gamma.p_nat, fa.rho.a, fa.rho.b, op);
       
        fa = compute_full_approximation_gamma(fa, si, pr);
    end

    %% sigma2 and (the associated) likelihood VB update
    if pr.sigma2_prior
        % sigma2 update
        si.y_lik = update_gaussian_lik_prec_site_vb(si.y_lik, fa.w.Tau_chol, fa.w.Mean, x, pr.yy, pr.xy, pr.xx, op);
        
        fa = compute_full_approximation_sigma2(fa, si, pr);

        % likelihood update
        si.y_lik = update_gaussian_lik_normal_site_vb(si.y_lik, fa.sigma2.imean, pr.xx, pr.xy);

        % full approx update
        fa = compute_full_approximation_w(fa, si, pr);
    end
    
    %% gamma feedback updates
    if n_gamma_feedbacks > 0
        % cavity
        ca_gf = compute_bernoulli_lik_cavity(fa.gamma.p_nat, si.gamma_feedback, gamma_feedbacks(:, 2));

        % moments of tilted dists
        ti_gf = compute_bernoulli_lik_tilt(ca_gf, pr, gamma_feedbacks(:, 1));

        % site updates
        si.gamma_feedback = update_bernoulli_lik_sites(si.gamma_feedback, ca_gf, ti_gf, op, gamma_feedbacks(:, 2));

        % full approx update (update only gamma part as only those sites have been updated)
        fa = compute_full_approximation_gamma(fa, si, pr);
    end

    %% show progress and check for convergence
    [converged, conv] = report_progress_and_check_convergence(conv, iter, z_w, fa, op);
    if converged
        if op.verbosity > 0
            fprintf(1, 'EP converged on iteration %d\n', iter);
        end
        break
    end
    
    %% update damp
    op.damp = op.damp * op.damp_decay;
end

if op.verbosity > 0 && converged == 0
    fprintf(1, 'EP hit maximum number of iterations\n');
end

if nargout > 3
    subfunctions.update_gaussian_lik_normal_site_vb = @update_gaussian_lik_normal_site_vb;
    subfunctions.update_gaussian_lik_prec_site_vb = @update_gaussian_lik_prec_site_vb;
    subfunctions.update_bernoulli_sites_vb = @update_bernoulli_sites_vb;
    subfunctions.update_bernoulli_sites_ep = @update_bernoulli_sites_ep;
    subfunctions.compute_bernoulli_lik_cavity = @compute_bernoulli_lik_cavity;
    subfunctions.compute_bernoulli_lik_tilt = @compute_bernoulli_lik_tilt;
    subfunctions.update_bernoulli_lik_sites = @update_bernoulli_lik_sites;
    subfunctions.compute_sns_prior_cavity = @compute_sns_prior_cavity;
    subfunctions.compute_sns_prior_tilt = @compute_sns_prior_tilt;
    subfunctions.update_sns_prior_sites = @update_sns_prior_sites;
    subfunctions.compute_full_approximation = @compute_full_approximation;
    subfunctions.compute_full_approximation_rho = @compute_full_approximation_rho;
    subfunctions.compute_full_approximation_sigma2 = @compute_full_approximation_sigma2;
    subfunctions.compute_full_approximation_w = @compute_full_approximation_w;
    subfunctions.compute_full_approximation_gamma = @compute_full_approximation_gamma;
end

end


function si = update_gaussian_lik_normal_site_vb(si, prec_mean, xx, xy)

si.normal_Tau = prec_mean * xx;
si.normal_Mu = prec_mean * xy;

end


function si = update_gaussian_lik_prec_site_vb(si, normal_Tau_chol, normal_Mean, x, yy, xy, xx, op)

tr_tmp = x / normal_Tau_chol';

si.gamma_b = (1 - op.damp) * si.gamma_b + op.damp * (0.5 * (yy - 2 * (normal_Mean' * xy) + tr_tmp(:)' * tr_tmp(:) + normal_Mean' * xx * normal_Mean));
%si.lik.sigma2.b = 0.5 * (pr.yy - 2 * (fa.w.Mean' * pr.xy) + tr_tmp(:)' * tr_tmp(:) + fa.w.Mean' * pr.xx * fa.w.Mean);

end


function si = update_bernoulli_sites_vb(si, p, op)
% This updates the conditioning variable (probability parameter).

si.beta_a = (1 - op.damp) * si.beta_a + op.damp * p;
si.beta_b = (1 - op.damp) * si.beta_b + op.damp * (1 - p);
%si.prior.rho.a = fa.gamma.p;
%si.prior.rho.b = (1 - fa.gamma.p);

end


function si = update_bernoulli_sites_ep(si, fa_bernoulli_p_nat, fa_beta_a, fa_beta_b, op)
% This updates the main variable (indicator variable).

% cavity
cav_nat = fa_bernoulli_p_nat - si.bernoulli_p_nat;
cav_a_m_cav_nat = (fa_beta_a - si.beta_a - 1 + eps) .* exp(cav_nat);
cav_b = fa_beta_b - si.beta_b - 1 + eps;

% tilt
ti_mean = cav_a_m_cav_nat ./ (cav_a_m_cav_nat + cav_b);
ti_mean = max(min(ti_mean, 1-eps), eps);

% site update
si.bernoulli_p_nat = (1 - op.damp) * si.bernoulli_p_nat + op.damp * (log(ti_mean) - log1p(-ti_mean) - cav_nat);

end


function ca = compute_bernoulli_lik_cavity(bernoulli_p_nat, si, inds)

if nargin < 3
    ca.bernoulli_p_nat = bernoulli_p_nat - si.bernoulli_p_nat;
else
    ca.bernoulli_p_nat = bernoulli_p_nat(inds) - si.bernoulli_p_nat(inds);
end

end


function ti = compute_bernoulli_lik_tilt(ca, pr, observations)

ti.bernoulli_mean = 1 ./ (1 + exp(-(ca.bernoulli_p_nat + (2 * observations - 1) .* pr.p_u_nat)));
ti.bernoulli_mean = max(min(ti.bernoulli_mean, 1-eps), eps);

end


function si = update_bernoulli_lik_sites(si, ca, ti, op, inds)

if nargin < 5
    si.bernoulli_p_nat = (1 - op.damp) * si.bernoulli_p_nat + op.damp * (log(ti.bernoulli_mean) - log1p(-ti.bernoulli_mean) - ca.bernoulli_p_nat);
else
    si.bernoulli_p_nat(inds) = (1 - op.damp) * si.bernoulli_p_nat(inds) + op.damp * (log(ti.bernoulli_mean) - log1p(-ti.bernoulli_mean) - ca.bernoulli_p_nat);
end

end


function ca = compute_sns_prior_cavity(fa, si, pr)

m = pr.m;

tmp = fa.w.Tau_chol \ eye(m);
var_w = sum(tmp.^2)';

denom = (1 - si.normal_tau .* var_w);
ca.normal_tau = denom ./ var_w;
ca.normal_mean = (fa.w.Mean - var_w .* si.normal_mu) ./ denom;

ca.bernoulli_p_nat = fa.gamma.p_nat - si.bernoulli_p_nat;
ca.bernoulli_p = 1 ./ (1 + exp(-ca.bernoulli_p_nat));

end


function [ti, z] = compute_sns_prior_tilt(ca, pr)

t = ca.normal_tau + 1 ./ pr.tau2;

g_var = 1 ./ ca.normal_tau; % for gamma0
mcav2 = ca.normal_mean.^2;
log_z_gamma0 = log1p(-ca.bernoulli_p) - 0.5 * log(g_var) - 0.5 * mcav2 ./ g_var;
g_var = pr.tau2 + g_var; % for gamma1
log_z_gamma1 = log(ca.bernoulli_p) - 0.5 * log(g_var) - 0.5 * mcav2 ./ g_var;
z_gamma0 = exp(log_z_gamma0 - log_z_gamma1);
z_gamma1 = ones(size(log_z_gamma1));
z = 1 + z_gamma0;

ti.normal_mean = z_gamma1 .* (ca.normal_tau .* ca.normal_mean) ./ t ./ z;
ti_normal_e2 = z_gamma1 .* (1 ./ t + 1 ./ t.^2 .* (ca.normal_tau .* ca.normal_mean).^2) ./ z;
ti.normal_var = ti_normal_e2 - ti.normal_mean.^2;

ti.bernoulli_mean = z_gamma1 ./ z;
ti.bernoulli_mean = max(min(ti.bernoulli_mean, 1-eps), eps);

end


function [si, nonpositive_cavity_vars, nonpositive_site_var_proposals] = update_sns_prior_sites(si, ca, ti, op)

nonpositive_site_var_proposals = false;

% skip negative cavs
update_inds = ca.normal_tau(:) > 0;
nonpositive_cavity_vars = ~all(update_inds);

new_tau_w_site = 1 ./ ti.normal_var - ca.normal_tau;

switch op.robust_updates
    case 0
    case 1
        inds_tmp = new_tau_w_site(:) > 0;
        nonpositive_site_var_proposals = ~all(inds_tmp);
        update_inds = update_inds & inds_tmp;
    case 2
        inds = new_tau_w_site(:) <= 0;
        new_tau_w_site(inds) = op.min_site_prec;
        ti.normal_var(inds) = 1./(op.min_site_prec + ca.normal_tau(inds));
end
new_mu_w_site = ti.normal_mean ./ ti.normal_var - ca.normal_tau .* ca.normal_mean;
si.normal_tau(update_inds) = (1 - op.damp) * si.normal_tau(update_inds) + op.damp * new_tau_w_site(update_inds);
si.normal_mu(update_inds) = (1 - op.damp) * si.normal_mu(update_inds) + op.damp * new_mu_w_site(update_inds);

si.bernoulli_p_nat(update_inds) = (1 - op.damp) * si.bernoulli_p_nat(update_inds) + op.damp * (log(ti.bernoulli_mean(update_inds)) - log1p(-ti.bernoulli_mean(update_inds)) - ca.bernoulli_p_nat(update_inds));

end


function fa = compute_full_approximation(si, pr)

fa = struct;
fa = compute_full_approximation_w(fa, si, pr);
fa = compute_full_approximation_gamma(fa, si, pr);
if pr.sigma2_prior
    fa = compute_full_approximation_sigma2(fa, si, pr);
end
if pr.rho_prior
    fa = compute_full_approximation_rho(fa, si, pr);
end

end


function fa = compute_full_approximation_rho(fa, si, pr)

% These are Beta distribution parameters in the common parametrization;
% pr params are also, while si params are natural parameters.
fa.rho.a = sum(si.gamma_prior.beta_a) + pr.rho_a;
fa.rho.b = sum(si.gamma_prior.beta_b) + pr.rho_b;

end


function fa = compute_full_approximation_sigma2(fa, si, pr)

% a and b are in the common parametrization of Gamma (the one with mean = a/b)
fa.sigma2.imean = (pr.sigma2_a + si.y_lik.gamma_a) / (pr.sigma2_b + si.y_lik.gamma_b); % note: approx is for sigma2^-1

end


function fa = compute_full_approximation_w(fa, si, pr)

% m x m and m x 1
fa.w.Tau = si.y_lik.normal_Tau + si.w_feedback.normal_Tau + diag(si.w_prior.normal_tau);
fa.w.Tau_chol = chol(fa.w.Tau, 'lower');
fa.w.Mu = si.y_lik.normal_Mu + si.w_feedback.normal_Mu + si.w_prior.normal_mu;
fa.w.Mean = fa.w.Tau_chol' \ (fa.w.Tau_chol \ fa.w.Mu);

end


function fa = compute_full_approximation_gamma(fa, si, pr)

fa.gamma.p_nat = si.w_prior.bernoulli_p_nat + si.gamma_feedback.bernoulli_p_nat + si.gamma_prior.bernoulli_p_nat;
fa.gamma.p = 1 ./ (1 + exp(-fa.gamma.p_nat));

end


function [converged, conv] = report_progress_and_check_convergence(conv, iter, z, fa, op)

conv_z = mean(abs(z(:) - conv.z_old(:)));
conv_P_gamma = mean(abs(fa.gamma.p(:) - conv.P_gamma_old(:)));

if op.verbosity > 0 && mod(iter, op.verbosity) == 0
    fprintf(1, '%d, conv = [%.2e %.2e], damp = %.2e\n', iter, conv_z, conv_P_gamma, op.damp);
end

%converged = conv_z < op.threshold && conv_P_gamma < op.threshold;
converged = conv_P_gamma < op.threshold;

conv.z_old = z;
conv.P_gamma_old = fa.gamma.p;

end