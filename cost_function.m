## Copyright (C) 2023 Horia Mercan

## Created: 2023-04-22

function [J, grad] = cost_function (params, X, y, lambda, input_layer_size, hidden_layer_size, output_layer_size)
    
    output_precision(16);
    first_final = hidden_layer_size * (input_layer_size + 1);
    Theta1 = reshape(params(1 : first_final), 
                      hidden_layer_size, input_layer_size + 1)';
                     
    Theta2 = reshape(params(first_final + 1 : end),
                      output_layer_size, hidden_layer_size + 1)';
    
    [classes, a2, z2, a1] = forward_propagation(X, params, 
                        input_layer_size, hidden_layer_size, output_layer_size);

    [y_ext] = extend_classes(y, output_layer_size);        
    delta3 = - y_ext + classes;
    
    [m, input_layer_size] = size(X);

    delta2 = a2 * delta3;

    [sigm_z2_der] = sigmoid_derivative(z2);
    
    aux_d2 = Theta2 * delta3';
    d2 = (aux_d2(2:end , :)) .* sigm_z2_der;
    
    
    delta1 = a1 * d2';

    gradient1 = delta1 / m;

    gradient2 = delta2 / m;
    aux_gradient2 = Theta2(2:end, :);
        
    gradient2(2:end, :) += lambda * aux_gradient2/ m;
    

    grad1 = reshape(gradient1', prod(size(gradient1)), 1);
    grad2 = reshape(gradient2, prod(size(gradient2)), 1);
    
    grad = [grad1; grad2];

    [J] = calculate_cost(params, classes, y_ext, lambda, Theta1, Theta2);
    
endfunction
