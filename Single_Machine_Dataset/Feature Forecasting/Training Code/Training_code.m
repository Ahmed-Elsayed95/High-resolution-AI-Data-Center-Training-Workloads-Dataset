%% load training data and adjust regression window 
Dataset=readtable('Weather_data.csv');

%%% meteorological Data
Data.Duration_years=1;                %  long of Utilized Data in years 
Data.time_step=15/60;                 % The data is recorded every 15 mintues. 

%%% Define the columns of the Data:
Data.P_PV=Dataset.EnergyDelta_Wh_(1:Data.Duration_years*365*24/Data.time_step);
Data.Gs=Dataset.GHI(1:Data.Duration_years*365*24/Data.time_step);
Data.T=Dataset.temp(1:Data.Duration_years*365*24/Data.time_step);
Data.Humidity=Dataset.humidity(1:Data.Duration_years*365*24/Data.time_step);
Data.V_Wind=Dataset.wind_speed(1:Data.Duration_years*365*24/Data.time_step);
Data.Press=Dataset.pressure(1:Data.Duration_years*365*24/Data.time_step);
%%% Weather Condition Data 
Data.rain=Dataset.rain_1h(1:Data.Duration_years*365*24/Data.time_step);
Data.snow=Dataset.snow_1h(1:Data.Duration_years*365*24/Data.time_step);
Data.weatherCond=Dataset.weather_type(1:Data.Duration_years*365*24/Data.time_step);
%% Create the input vector 
X=[Data.P_PV Data.Gs Data.T Data.Humidity Data.Press Data.V_Wind Data.rain Data.snow Data.weatherCond];

%% Build the training model 
Win_size=2*24/Data.time_step;                   % Set the input window length  
Forcast_horizion=1;                           % output regresion size 

% 1) Create sequences for prediction (X = history, Y = next timestep)
input_data = cell(length(Data.P_PV)-Win_size-Forcast_horizion, 1);              % historical samples as cell of sequance 
output_data = cell(length(Data.P_PV)-Win_size-Forcast_horizion, 1);               % prediction samples as next layer sequance

% 2) Create the Dataset for trining, validation and testing  
for i = 1:length(Data.P_PV)-Win_size-Forcast_horizion
    % Input: 48 hours of history
    input_data{i} = X(i:i-1+Win_size,:);
    % Target: Next timestep's features (healthy state only)
    output_data{i} = Data.P_PV(i+Forcast_horizion:i+Win_size+Forcast_horizion-1);
end

input_size=length(input_data{1}(1,:));
output_size=length(output_data{1}(1,:));
Samples_length=length(output_data);

% 3) Split into train, validation, test sets (70/15/15 split)
rng(42);                                        % For reproducibility
idx = randperm(Samples_length);
numTrain = floor(0.7 * Samples_length);
numVal = floor(0.15 * Samples_length);
XTrain_LSTM = input_data(idx(1:numTrain));
YTrain_LSTM = output_data(idx(1:numTrain));
XVal_LSTM = input_data(idx(numTrain+1:numTrain+numVal));
YVal_LSTM = output_data(idx(numTrain+1:numTrain+numVal));
XTest_LSTM = input_data(idx(numTrain+numVal+1:end));
YTest_LSTM = output_data(idx(numTrain+numVal+1:end));

% 4) Build LSTM structure, training options and training
% Define LSTM regression network
    layers = [
        sequenceInputLayer(input_size,"Normalization","zscore",MinLength=Win_size)
        convolution1dLayer(1,96)
        lstmLayer(1*250)
        fullyConnectedLayer(1*500)
        reluLayer
        fullyConnectedLayer(output_size)];
    
    % 5) Training options and Train Network 
    options = trainingOptions('adam', ...
        'MaxEpochs', 350, ...
        'MiniBatchSize', 100, ...
        'Shuffle','every-epoch',...
        'ValidationData', {XVal_LSTM, YVal_LSTM}, ...
        'ValidationFrequency', 30, ...
        'Plots', 'training-progress', ...
        'Verbose', true, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.9, ...
        'LearnRateDropPeriod', 15,...
        'ExecutionEnvironment','gpu');

    analyzeNetwork(layers)
    
    %5) Train network
    [net_LSTM, TrainInfo_LSTM] = trainnet(XTrain_LSTM,YTrain_LSTM,layers,"mse",options);
    
    
    
    % 6) Evaluation of the network
    % 4- Testing and validation:
    YPred = minibatchpredict(net_LSTM,XTest_LSTM);
    YPred=squeeze(num2cell(YPred, [1 2]));
    for n = 1:length(XTest_LSTM)
      err_LSTM(n) = rmse(YPred{n}(end),YTest_LSTM{n}(end),"all");
    end
    figure
    histogram(err_LSTM)
    xlabel("RMSE")
    ylabel("P.u")
    mean(err_LSTM)
    max(err_LSTM)
    min(err_LSTM)
    
    % Get whole Data-set test
    YPred = minibatchpredict(net_LSTM,input_data);
    YPred=squeeze(num2cell(YPred, [1 2]));
    Y_act=[];
    Y_hat_LSTM=[];
    for t=1:1:length(YPred)
    Y_act(t)=output_data{t}(end);
    Y_hat_LSTM(t)=YPred{t}(end);
    end
    Y_hat_LSTM(abs(Y_hat_LSTM)<0.1)=0;     % filter data:
    figure;
    plot([Y_act', Y_hat_LSTM']);

    %%% Save name
    Single_machine_EmbedDim256_HiddenUnit512_batchsize128