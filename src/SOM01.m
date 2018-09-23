%% Load the data
input = normalisation_std(xlsread('inputData.xlsx'));
output = normalisation_std(xlsread('outputData.xlsx'));
%% Remove some indicators that have a strong correlation
input = input([1:4,7:12,14:17],:);
input = [input;output];
time_periods = 36;
indicator_number = size(input,1);
%% Reshape the data in a per-country basis
input_per_country = reshape(input, [indicator_number,time_periods,size(input,2)/time_periods]);
countries = [string('Spain'),string('France'),string('Germany'),...
            string('Cuba'),string('Dominican Republic'),string('Mexico'),...
            string('Colombia'),string('Venzuela'),string('Argentina'),...
            string('Uruguay'),string('Chile'),string('Bolivia'),...
            string('Puerto Rico'),string('Paraguay'),string('El Salvador'),...
            string('Honduras'),string('Ecuador'),string('Peru'),...
            string('Panama')];
%% Take Germany and France out of the data
input_final = [];
for i=1:length(countries)
    if countries(i) == 'Germany' || countries(i) == 'France'
        continue
    end
    input_final = [input_final input_per_country(:,:,i)];
    
end
        
%% Create the Self-Organizing Map
dimension1 = 20;
dimension2 = 20;
net = newsom(input_final,[dimension1 dimension2]);
% Use hexagonal topology
%%
net.adaptFcn = 'learnsom';
net.trainFcn = 'trainr';
net.trainParam.epochs = 50000;
%% Train the Network
[net,tr] = train(net,input);
%% Plot the activated neurons for each country
for i = 1:length(countries)
   figure, plotsomhits(net,input_per_country(:,:,i));
end

