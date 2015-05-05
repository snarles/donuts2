%qlogin -l h_vmem=8g

rng(0)
n = 30; p = 50; q = 5; r = 1;
X = abs(randn(n, p));
B0 = abs(randn(p, q)) .* binornd(1, 0.1, p, q);
Y0 = X * B0;
U0 = rand(n, r);
V0 = rand(r, q);
E0 = U0 * V0;
err = randn(n, q);
mult = 10;
sigma = 2;
Y = Y0 + mult * E0 + sigma .* err;

norm(B0, 'fro')     % 4.05
norm(E0, 'fro')     % 3.87
norm_nuc(E0)        % 3.87
%% solve with NNLS

cvx_begin
  variable B(p, q)
  minimize( norm(Y - X * B, 'fro') )
  subject to
    B >= 0
cvx_end
norm(B - B0, 'fro')

% mult norm
% 0.0  3.83
% 2    4.15
% 5    4.58
% 10   5.72

%% solve with NNLS + factor

nncons = 5

cvx_begin
  variable B(p, q)
  variable E(n, q)
  minimize( norm(Y - X * B - E, 'fro') )
  subject to
    B >= 0
    norm_nuc(E) <= nncons
cvx_end

norm(B - B0, 'fro')

% mult nncons norm_err
% 5    10      4.68   
%      3       4.59   
% 10   10      5.68
%      



