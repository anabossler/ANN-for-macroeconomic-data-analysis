function result = RNNPrediction(traininput, trainoutput, trainAlg, transfFunc, layers, maxFail, maxTime, maxEpochs, minError, alpha, n, k, confidence)
    % Loading and preprocessing data
    input = normalisation_std(xlsread(traininput));
    output = normalisation_std(xlsread(trainoutput));
    input = con2seq(input);
    output = con2seq(output);
    % Global statistics
    trainPerfG = zeros(1,n);
    validPerfG = zeros(1,n);
    testPerfG = zeros(1,n);
    epochsG = zeros(1,n);
    timesG = zeros(1,n);
    % Start iterations 
    for j = 1:n
        foldSize = size(input,2)/k;
        % Local statistics for every fold
        trainPerf = zeros(1,k);
        validPerf = zeros(1,k);
        testPerf = zeros(1,k);
        epochs = zeros(1,k);
        times = zeros(1,k);
        sequenceLength = 5;
        % Perform the folds
        for i = 1:k
            % Initialize and configure the RNN
            rnn = layrecnet(1:sequenceLength,layers,trainAlg);
            rnn.trainParam.time = maxTime;
            rnn.trainParam.epochs = maxEpochs;
            rnn.trainParam.lr = alpha;
            rnn.trainParam.goal = minError;
            % Divide data by indices
            rnn.divideFcn = 'divideind';
            % Divide the data into folds
            % Set training data
            if(i == 1)
                rnn.divideParam.trainInd = [(i*foldSize + 1): k*foldSize];
                trainInd = [(i*foldSize + 1): k*foldSize];
            elseif (i == k)
                rnn.divideParam.trainInd = [1: ((i-1)*foldSize)];
                trainInd = [1: ((i-1)*foldSize)];
            else
                rnn.divideParam.trainInd = [1: ((i-1)*foldSize),(i*foldSize + 1):k*foldSize];
                trainInd = [1: ((i-1)*foldSize),(i*foldSize + 1):k*foldSize];
            end
            kthFold = [((i-1)*foldSize + 1):(i*foldSize)];
            half = ceil(length(kthFold)/2);
            % Set testing data
            rnn.divideParam.testInd = kthFold(1:half);
            % Set validation data
            rnn.divideParam.valInd = kthFold((half+1):length(kthFold));
            testInd = kthFold(1:half);
            valInd = kthFold((half+1):length(kthFold));
            % Train the RNN
            [Xs,Xi,Ai,Ts] = preparets(rnn,input(trainInd),output(trainInd));
            [rnn, tr] = train(rnn, Xs,Ts, Xi,Ai);
            % Compute the output for the test
            [Xs,Xi,Ai,Ts] = preparets(rnn,input(testInd),output(testInd));
            rnnTestOutput = rnn(Xs,Xi,Ai);
            m2errTest = meanSquaredError(cell2mat(rnnTestOutput),cell2mat(output(testInd((sequenceLength+1):length(testInd)))));
            % Compute the output for the validation
            [Xs,Xi,Ai,Ts] = preparets(rnn,input(valInd),output(valInd));
            rnnValOutput = rnn(Xs,Xi,Ai);
            m2errVal = meanSquaredError(cell2mat(rnnValOutput),cell2mat(output(testInd((sequenceLength+2):length(testInd)))));
    
            validPerf(i) = m2errVal;
            trainPerf(i) = tr.best_perf;
            testPerf(i) = m2errTest;
            epochs(i) = tr.num_epochs;
            times(i) = tr.time(length(tr.time));
        end
        % Compute global stats
        trainPerfG(j) = mean(trainPerf);
        validPerfG(j) = mean(validPerf);
        testPerfG(j) = mean(testPerf);
        epochsG(j) = mean(epochs);
        timesG(j) = mean(times);
    end
    % a and z are used for the computation of confidence intervals
    a = (100+confidence)/2/100;
    z = norminv(a);
    % Prepare stats matrix
    stats = [mean(trainPerfG),mean(validPerfG),...
        mean(testPerfG),mean(epochsG),mean(timesG),std(testPerfG)*z/sqrt(n)];
    % Write stats matrix to a file
    csvwrite(strcat("RNN",trainAlg,transfFunc,mat2str(layers),'.csv'),stats);
end