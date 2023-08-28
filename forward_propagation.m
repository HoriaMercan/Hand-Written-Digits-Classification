## Copyright (C) 2023 Horia Mercan

## Created: 2023-04-29

function [classes, operation_mat2, z2, a1] = forward_propagation (X, weights, input_layer_size, ...
                                      hidden_layer_size, output_layer_size)
                 
    first_final = hidden_layer_size * (input_layer_size + 1);
    
    Theta1 = reshape(weights(1 : first_final), 
                      input_layer_size + 1, hidden_layer_size)';

    Theta2 = reshape(weights(first_final + 1 : end),
                      hidden_layer_size + 1, output_layer_size)';

    [m, input_layer_size] = size(X);
    operation_mat1 = ones(m, input_layer_size + 1);
    operation_mat1(1 : m, 2 : (input_layer_size + 1)) = X;

    operation_mat1 = operation_mat1';
    a1 = operation_mat1;
    before_sigm1 = Theta1 * operation_mat1;
    z2 = before_sigm1;

    [ans_1] = sigmoid(z2);

    operation_mat2 = ones(hidden_layer_size + 1, m);
    operation_mat2 (2 : (hidden_layer_size + 1), 1:m) = ans_1;

    a2 = zeros(m, hidden_layer_size);
    before_a2 = Theta2 * operation_mat2;

    [a2] = sigmoid(before_a2);
    
    classes = a2';

endfunction
