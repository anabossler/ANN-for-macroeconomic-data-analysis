% Set the ranges and number of seeds
a = 0;
b = 10;
seedsNumber = 5;
% Red, Green, Blue, Magenta, Orange
seedsColors = [1 0 0; 0 1 0; 0 0 1; 1 0 1; 1 0.5 0];
% Place the seeds randomly in the range [a,b]x[a,b]
seedsX = (b-a).* rand(1, seedsNumber) + a;
seedsY = (b-a).* rand(1, seedsNumber) + a;
% Set the number of points per seed and their matrices
pointSize = 30;
pointsPerSeed = 50;
pointsX = zeros(1,seedsNumber*pointsPerSeed);
pointsY = zeros(1,seedsNumber*pointsPerSeed);
pointsColors = zeros(seedsNumber*pointsPerSeed,3);
% Place the points around the seeds
for i = 1:seedsNumber 
    for j = 1:pointsPerSeed
       pointsX(((i-1)*pointsPerSeed) + j) = normrnd(0,0.5) + seedsX(i);
       pointsY(((i-1)*pointsPerSeed) + j) = normrnd(0,0.5) + seedsY(i);
       pointsColors(((i-1)*pointsPerSeed) + j,:) = seedsColors(i,:);
    end
end
% Plot the points
figure
title('2D data points')
scatter(pointsX,pointsY,10,pointsColors,'filled');
hold on
% Run fuzzy C-means clustering on the points
[centers, U] = fcm([pointsX; pointsY]',seedsNumber);
% Plot the points from the fuzzy C-means clustering algorithm
scatter(centers(:,1),centers(:,2),50, [0 0 0], 'filled');
%% Self-Organization Map
%Prepare the data for the SOM 
x = [pointsX; pointsY];
% Create and set dimensions of the SOM
dimension1 = 5;
dimension2 = 5;
net = selforgmap([dimension1 dimension2]);
net.layers{1}.topologyFcn = 'hextop';
net.trainParam.epochs = 1000;
% Train the SOM
[net,tr] = train(net,x);
% Plot the activated neurons for each cluster
figure, plotsomhits(net,x(:,1:pointsPerCluster)),title('Red cluster');
figure, plotsomhits(net,x(:,(pointsPerCluster + 1):pointsPerCluster*2)),title('Green cluster');
figure, plotsomhits(net,x(:,(pointsPerCluster*2 + 1):pointsPerCluster*3)),title('Blue cluster');
figure, plotsomhits(net,x(:,(pointsPerCluster*3 + 1):pointsPerCluster*4)),title('Magenta cluster');
figure, plotsomhits(net,x(:,(pointsPerCluster*4 + 1):pointsPerCluster*5)),title('Orange cluster');