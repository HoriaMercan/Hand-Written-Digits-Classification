## Copyright (C) 2023 Horia Mercan

## Created: 2023-04-22

function [classes] = predict_classes (X, weights, input_layer_size,
                                      hidden_layer_size, output_layer_size)
                 
    [classes, operation_mat2, z2, a1] = forward_propagation (X, weights, input_layer_size, hidden_layer_size, output_layer_size)

endfunction
