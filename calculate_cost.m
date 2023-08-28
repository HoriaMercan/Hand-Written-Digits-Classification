## Copyright (C) 2023 Horia Mercan

## Created: 2023-05-03

function [J] = calculate_cost(params, classes, y, lambda, Theta1, Theta2)

    J = 0;
    [m, n] = size(params);

    [m, k] = size(y);
    
    for i = 1 : m
        for j = 1 : k
            J = J  - y(i, j) * log(classes(i, j))/m - (1 - y(i, j)) * log(1 - classes(i, j))/m;
        endfor
    endfor

    grad1 = reshape(Theta1, prod(size(Theta1)), 1);
    grad2 = reshape(Theta2, prod(size(Theta2)), 1);
    J = J + ((grad1' * grad1) + (grad2' * grad2)) * lambda / (2 * m);
endfunction