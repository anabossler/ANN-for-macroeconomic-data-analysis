%% Generating data points
x = linspace(-2*pi, 2*pi, 500);
y = sin(x);
xextended = linspace(-4*pi,4*pi,1000);
yextended = sin(xextended);

%% Create and configure the MLP
layer = [50 10];
% gradient descend (backpropagation) algorithm will be used
net = feedforwardnet(layer,'traingd');
net = configure(net,x,y);
net = init(net);
        
for h = 1:length(layer)
    %Using sigmoid transfer function
    net.layers{h}.transferFcn = 'tansig'; 
end

net.trainParam.max_fail = 150;
net.trainParam.time = 600;
net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-6;
net.trainParam.lr = 0.01;
%% Train the MLP
[net,tr] = train(net,x,y);
%% Generate the output of the MLP
outputs = net(x);
outputsextended = net(xextended);
%% Create and configure the RNN
% The RNN will use an input sequence of length 100
% gradient descend (backpropagation) algorithm will be used
rnn = layrecnet(1:100,[15],'traingd');
rnn.trainParam.epochs = 2000;
rnn.trainParam.lr = 0.01;
%% Transforming the training data to fit the RNN
[Xs,Xi,Ai,Ts] = preparets(rnn,num2cell(x),num2cell(y));
%% Training the RNN
rnn = train(rnn,Xs,Ts,Xi,Ai);
%% Transforming the testing data to fit the RNN
[Xs2,Xi2,Ai2,Ts2] = preparets(rnn,num2cell(xextended),num2cell(yextended));
%% Generate the output of the RNN
rnnoutput = rnn(Xs2,Xi2,Ai2);
%% Transform the output of the RNN
rnnoutput = cell2mat(rnnoutput);

%% Generate the plot for the MLP
figure 
plot(xextended(101:(length(xextended))),yextended(101:(length(yextended))),'Color',[1 0 0], 'LineWidth',2)
hold on
plot(xextended(101:(length(xextended))),outputsextended(101:length(outputsextended)),'Color', [0 1 0], 'LineWidth',2);
line([-2*pi -2*pi], [-2 2], 'Color', [0 0 0]);
line([2*pi 2*pi], [-2 2], 'Color', [0 0 0]);
xlabel('-4\pi < x < 4\pi'); % x-axis label
ylabel('sine function approximation');
legend('y = sin(x)', 'MLP approximation');

%% Generate the plot for the RNN
figure 
plot(xextended(101:(length(xextended))),yextended(101:(length(yextended))),'Color',[1 0 0], 'LineWidth',2)
hold on
plot(xextended(101:(length(xextended))),rnnoutput, 'Color', [0 0 1], 'LineWidth',2)
line([-2*pi -2*pi], [-2 2], 'Color', [0 0 0]);
line([2*pi 2*pi], [-2 2], 'Color', [0 0 0]);
xlabel('-4\pi < x < 4\pi'); % x-axis label
ylabel('sine function approximation');
legend('y = sin(x)', 'RNN approximation');