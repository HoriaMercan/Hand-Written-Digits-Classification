## Copyright (C) 2023 Horia Mercan

## Created: 2023-04-22

function [matrix] = initialize_weights (L_prev, L_next)
    # initializam o matrice cu valori aleatorii in intervalul (-1, 1)
    matrix = 2 * rand(L_next, L_prev + 1) - ones(L_next, L_prev + 1);
    
    eps = sqrt(6) / sqrt(L_prev + L_next);
    
    matrix = eps * matrix;
endfunction
