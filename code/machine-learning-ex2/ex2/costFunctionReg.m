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
J_1= 0;
J_2= 0;
for i=1:m
  J_1= -y(i)*log(sigmoid(X(i,:)*theta))-(1-y(i))*log(1-sigmoid(X(i,:)*theta))+J_1;
endfor
J_1 = J_1/m;
for i=2:size(theta)
  J_2= J_2+theta(i)*theta(i);
endfor
J_2 = J_2*lambda/(2*m);
J = J_1+J_2;
for i=1:m
  for k=1:size(theta)
    grad(k)=grad(k)+(sigmoid(X(i,:)*theta)-y(i))*X(i,k);
  end
end
grad = grad./m;
for i=2:size(theta)
  grad(i) = grad(i)+lambda/m*theta(i);
end





% =============================================================

end