function result = GDPPrediction(traininput, trainoutput,trainAlg,transfFunc,layers,maxFail,maxTime,maxEpochs,minError,alpha,n,k,confidence)
    % Loading data
    input = normalisation_std(xlsread(traininput));
    output = normalisation_std(xlsread(trainoutput));
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
        % Perform the folds
        for i = 1:k
            % Initialize and configure the MLP
            net = feedforwardnet(layers,trainAlg);
            net = configure(net,input,output);
            net = init(net);
            for h = 1:length(layers)
                net.layers{h}.transferFcn = transfFunc; 
            end
            net.trainParam.max_fail = maxFail;
            net.trainParam.time = maxTime;
            net.trainParam.epochs = maxEpochs;
            net.trainParam.goal = minError;
            net.trainParam.lr = alpha;
            % Divide data by index
            net.divideFcn = 'divideind';
            % Divide the data into folds
            % Set training data
            if(i == 1)
                net.divideParam.trainInd = [(i*foldSize + 1): k*foldSize];
            elseif (i == k)
                net.divideParam.trainInd = [1: ((i-1)*foldSize)];
            else
                net.divideParam.trainInd = [1: ((i-1)*foldSize),(i*foldSize + 1):k*foldSize];
            end 
            kthFold = [((i-1)*foldSize + 1):(i*foldSize)];
            half = ceil(length(kthFold)/2);
            % Set testing data
            net.divideParam.testInd = kthFold(1:half);
            % Set validation data
            net.divideParam.valInd = kthFold((half+1):length(kthFold));
            % Train the MLP
            [net,tr] = train(net,input,output);
            % Store local stats
            validPerf(i) = tr.best_vperf;
            trainPerf(i) = tr.best_perf;
            testPerf(i) = tr.best_tperf;
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
    csvwrite(strcat("MLP",trainAlg,transfFunc,mat2str(layers),'.csv'),stats);
end