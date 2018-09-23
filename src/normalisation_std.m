function vectorNormalized = normalisation_std(matrix)
    
    numRows = size(matrix,1);
    numCols = size(matrix,2);
    vectorNormalized = zeros(numRows,numCols);
    
    for i = 1:numRows
        vector = matrix(i,:);
        
        meanValue = mean(vector);
        standard = std(vector);
        vectorNormalized(i,:) = (vector - meanValue)/(standard);
    end
    
    
end

