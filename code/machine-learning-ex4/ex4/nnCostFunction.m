function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
X = [ones(m,1),X];%add bias
layer_2 = sigmoid(X*transpose(Theta1));
layer_2 = [ones(size(layer_2,1),1),layer_2];
layer_3 = sigmoid(layer_2*transpose(Theta2));
regulation_term = 0;
label_y = zeros(m,num_labels);

for i = 1:m
    
    label_y(i,y(i))=1;
    for k = 1:num_labels
        J = J + (-label_y(i,k)*log(layer_3(i,k)))-((1-label_y(i,k))*log(1-layer_3(i,k)));
    end    
end

for i = 1:size(Theta1,1)
    for k  = 2:size(Theta1,2) %no bias term
        regulation_term =regulation_term + Theta1(i,k)^2;
    end
end
for i = 1:size(Theta2,1)
    for k  = 2:size(Theta2,2)
        regulation_term =regulation_term+ Theta2(i,k)^2;
    end
end
J =J/m;
regulation_term = regulation_term*lambda/(2*m);
J = J+regulation_term;
dleta_2 = 0;
dleta_1 = 0;

for i = 1:m
    layer_2 = sigmoid(X(i,:)*transpose(Theta1));
    layer_2 = [ones(size(layer_2,1),1),layer_2];
    layer_3 = sigmoid(layer_2*transpose(Theta2));
    vector_y = zeros(1,num_labels);
    vector_y(1,y(i,:)) = 1;
    error_3 = layer_3-vector_y(1,:);
    error_3 = transpose(error_3);
    error_2 = transpose(Theta2)*error_3.*transpose(layer_2.*(1-layer_2));
    dleta_2 = dleta_2+error_3*layer_2;
    error_2 = error_2(2:end,:);
    %error_1 = transpose(Theta1)*error_2.*transpose(sigmoidGradient(X(i,:)));
    dleta_1 = dleta_1+error_2*X(i,:);    
end
Theta2_grad = dleta_2/m;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end)+(lambda/m)*Theta2(:,2:end); 
Theta1_grad = dleta_1/m;
Theta1_grad(:,2:end) = Theta1_grad(:,2:end)+(lambda/m)*Theta1(:,2:end);
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
