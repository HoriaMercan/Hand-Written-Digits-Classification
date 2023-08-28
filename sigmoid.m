## Copyright (C) 2023 Horia Mercan

## Created: 2023-04-21

function [ans] = sigmoid (x)
  [lines, colls] = size(x);
  #ans = 1 / (1 + e ^ (-x));
  
  ans = zeros(lines, colls);
  for i = 1 : lines
    for j = 1 : colls
      ans(i, j) = 1 / (1 + e ^ (-x(i, j)));
    endfor
  endfor
endfunction
