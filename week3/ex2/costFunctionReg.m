function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

hypothesis = 1 ./ (1 + exp(-1 * X * theta));
thetaZero = theta;
thetaZero(1) = 0;

J = (1/m) .* sum(-1 * y .* log(hypothesis) -1 * (1-y) .* log(1 - hypothesis)) + (lambda/(2*m) * sum(thetaZero .^2));		
	 

errors = hypothesis - y;
for j = 1:length(grad),
	if j == 1,
		grad(j) = (1/m) .* sum(errors .* X(:, j));
	else,
		grad(j) = (1/m) .* sum(errors .* X(:, j)) + (lambda*theta(j))/m;
	end

end





% =============================================================

end
