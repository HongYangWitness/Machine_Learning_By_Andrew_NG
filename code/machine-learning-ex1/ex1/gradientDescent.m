function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    h = [m,1];
    difference = [0,0];
    for k= 1:m
      h(k)=X(k,2)*theta(2)+theta(1);
    end
    for k= 1:m
      difference(1) = difference(1) +(h(k)-y(k));
      difference(2) = difference(2) +(h(k)-y(k))*X(k,2);
    end
    theta(1) = theta(1)- (alpha*difference(1))/m;
    theta(2) = theta(2)- (alpha*difference(2))/m;
    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

end


end
