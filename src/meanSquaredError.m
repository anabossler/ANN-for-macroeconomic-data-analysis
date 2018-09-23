function mse = meanSquaredError(inputArg1,inputArg2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

tmp = (inputArg1 - inputArg2).*(inputArg1 - inputArg2);
mse = sum(tmp)/length(tmp);

end

