function vectorNormalized = normalisation_tanh(matrix)

    numRows = size(matrix,1);
    numCols = size(matrix,2);
    vectorNormalized = zeros(numRows,numCols);


    for i = 1:numRows
        vector = matrix(i,:);
        
        meanValue = mean(vector);
        standard = std(vector);
        vectorNormalized(i,:) = (1/2)*(tanh(0.01*((vector - meanValue)/standard))+1);
    end
end

