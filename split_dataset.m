## Copyright (C) 2023 Horia Mercan

## Created: 2023-04-21

function [x_train, y_train, X_test, Y_test] = split_dataset (X, y, percent)
    [lines, columns] = size(X);
    X1 = zeros(lines, columns);
    y1 = zeros(lines, 1);
    
    perm = randperm(lines);
    
    for i = 1 : lines;
        X1(i, 1 : columns) = X(perm(i), 1 : columns);
        y1(i, 1) = y(perm(i));
    endfor
    
    number = floor(percent * lines);
    x_train = X1(1 : number, 1 : columns);
    y_train = y1(1 : number, 1);
    
    X_test = X1(number + 1 : lines, 1 : columns);
    Y_test = y1(number + 1 : lines, 1);
endfunction
