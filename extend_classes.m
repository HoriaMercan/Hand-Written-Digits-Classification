## Copyright (C) 2023 Horia Mercan

## Created: 2023-04-22

function [y_ext] = extend_classes(y, output_layer_size)
    [lines, colls] = size(y);
    y_ext = zeros(lines, output_layer_size);
    for i = 1 : lines
        y_ext(i, y(i)) = 1;
    endfor
endfunction
