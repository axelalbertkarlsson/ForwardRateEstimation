function [c,a, A] = loglikelihood(var, mu, sigma, dist, nu)
% ska vara samma dimensioner på var och mu, sigma samma fast kvadratisk
% matris
c = 0;
a = 0;
A = 0;
n = length(var);

if dist == "normal"
    % Cholesky decomposition to compute log-det
    %L = chol(sigma, 'lower'); % Cholesky decomposition
    %logDetSigma = 2 * sum(log(diag(L))); % Log-determinant
    %D_log = -0.5 * (logDetSigma + n * log(2 * pi));
    D = 1/( (2*pi)^n*det(sigma))^(1/2);
    c = log(D) + (-1/2 * (var - mu)' * (sigma \ (var - mu)));
    a = - sigma \ (var-mu);
    A = - inv(sigma);

elseif dist == "student"
    gamma1 = 1;
    gamma2 = 2;
    C = gamma1/(gamma2 * (nu*pi)^(n/2) * det(sigma)^(1/2) );
    c = log(C) -(nu +n)/2 *  log(1 + 1/nu * (var - mu)' * sigma \ (var-mu) );
    a = -(nu +n)/2 * ((2/nu * sigma \ (var - mu) ) / (1 + 1/nu * (var - mu)' * sigma \ (var-mu)  ));
    A = -(nu +n)/2 * ((4/nu^2 * sigma \ (var - mu) * (var-mu)' / sigma )  / (1 + 1/nu * (var - mu)' * sigma \ (var-mu)  )^2 );
else
    disp("not supported distribution")
end


end