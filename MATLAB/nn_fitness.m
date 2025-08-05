function [loss, metrics] = nn_fitness(hyperparams, returnMetrics)
    if nargin < 2, returnMetrics = false; end

    % Extract hyperparameters
    lr = hyperparams(1);
    neurons = round(hyperparams(2));
    momentum = hyperparams(3);

    % Load data
    data = readtable('processed_heat_stress_data_UNSCALED.csv');
    try
        X = data{:, {'rt', 'pr', 'htc', 'breed_CB', 'breed_NATIVE', 'breed_PHILDAIRY', ...
                     'temperature', 'relativeHumidity', 'season_Summer'}}';
        Y = data.rr';
    catch
        error('Check that all variable names exist in the CSV header.');
    end

    valid = all(~isnan(X), 1) & ~isnan(Y);
    X = X(:, valid);
    Y = Y(:, valid);

    net = fitnet(neurons);
    net.trainParam.lr = lr;
    net.trainParam.mc = momentum;
    net.trainParam.showWindow = false;
    net.divideParam.trainRatio = 0.7;
    net.divideParam.valRatio = 0.2;
    net.divideParam.testRatio = 0.1;

    try
        [net, tr] = train(net, X, Y);
        Ypred = net(X(:, tr.testInd));
        Ytrue = Y(:, tr.testInd);
        loss = perform(net, Ytrue, Ypred);

        rmse = sqrt(mean((Ytrue - Ypred).^2));
        mae = mean(abs(Ytrue - Ypred));
        r2 = 1 - sum((Ytrue - Ypred).^2) / sum((Ytrue - mean(Ytrue)).^2);
        metrics = struct('rmse', rmse, 'mae', mae, 'r2', r2);
    catch
        loss = 1e3;
        metrics = struct('rmse', NaN, 'mae', NaN, 'r2', NaN);
    end
end
