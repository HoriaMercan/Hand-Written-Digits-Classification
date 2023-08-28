## Copyright (C) 2023 Horia Mercan

## Created: 2023-04-28

function [z] = sigmoid_derivative (x)
  [lines, colls] = size(x);
  [ans1] = sigmoid(x);
  z = zeros(lines, colls);
  z = ans1 - ans1 .^2;
endfunction
