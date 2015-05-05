%qlogin -l h_vmem=8g

rng(0)

results = zeros(100, 5);

for i=1:100;
    n = 30; p = 50; q = 10; r = 1;
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

    yerr_0 = norm(Y - Y0, 'fro') % 70.43
    norm(Y.^2 - Y0.^2, 'fro') % 1e3
    norm(B0, 'fro')     % 7.50
    norm_nuc(E0)        % 5.76
    %% solve with NNLS

    cvx_begin
	variable B(p, q)
    	minimize( norm(Y - X * B, 'fro') )
    	subject to
    		B >= 0
    cvx_end

    berr_n = norm(B - B0, 'fro')
    yerr_n = norm(X * B - Y0, 'fro') % 60.14
    norm((X * B).^2 - (Y0).^2, 'fro') % 794.79

    % mult norm
    % 10   8.33

    %% solve with NNLS + factor

    nncons = 10

    cvx_begin
	variable B(p, q)
  	variable E(n, q)
  	minimize( norm(Y - X * B - E, 'fro') )
  	subject to
    		B >= 0
    		norm_nuc(E) <= nncons
    cvx_end

    berr_f = norm(B - B0, 'fro')
    yerr_f = norm(X * B - Y0, 'fro') % 60.57
    norm((X * B).^2 - (Y0).^2, 'fro') % 803.6

    % mult nncons norm_err
    % 10   5      8.21  
    %      10     8.15     

    %% results

    {yerr_0, yerr_n, yerr_f}

    {berr_n, berr_f}
    results(i, :) = [yerr_0, yerr_n, yerr_f, berr_n, berr_f];
end;

% >> mean(results)
%   67.4000   58.6607   59.3372    8.6837    8.4379


%% now to use TFOCS

addpath TFOCS

