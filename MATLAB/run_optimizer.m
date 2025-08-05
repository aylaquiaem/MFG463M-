clc; clear; close all;

% Define hyperparameter bounds
bounds = [
    0.001, 0.1;   % learning rate
    5,     100;   % hidden neurons
    0.5,   0.99   % momentum
];
dim = size(bounds, 1);

% Fitness function wrapper
func = @(x) nn_fitness(x, true);  % 'true' flag enables metrics output

iters = 30;
aveFit = zeros(1, iters);
results = zeros(iters, 8); % Preallocate for speed

doplot = true;

for i = 1:iters
    tic;
    [sol, fit, hist] = HHOPSO(@(x) nn_fitness(x, true), dim, bounds, 100, 20, 0.4, 2.05, 2.05, i, false);
    trainTime = toc;

    [~, metrics] = nn_fitness(sol, true);  % Evaluate metrics on best solution

    aveFit(i) = fit;

    fprintf('Iteration %d\n', i);
    fprintf('Best hyperparameters: %s\n', mat2str(sol, 4));
    fprintf('Best validation loss: %.10f\n', fit);
    fprintf('RMSE: %.4f | MAE: %.4f | R^2: %.4f\n', metrics.rmse, metrics.mae, metrics.r2);
    fprintf('Training time: %.2f seconds\n\n', trainTime);

    % Store all trial results
    results(i, :) = [sol, fit, metrics.rmse, metrics.mae, metrics.r2, trainTime];

    if doplot
        figure;
        plot(hist, 'b-', 'LineWidth', 2);
        xlabel('Iteration'); ylabel('Fitness (Validation Loss)');
        title(sprintf('Fitness over Time [Trial %d]', i));
        grid on;
    end
end

% Final Report (display)
fprintf('\nFinal Report:\n');
fprintf('Best fitness: %.10f\n', min(aveFit));
fprintf('Worst fitness: %.10f\n', max(aveFit));
fprintf('Median fitness: %.10f\n', median(aveFit));
fprintf('Mean fitness: %.10f\n', mean(aveFit));
fprintf('StDev fitness: %.10f\n', std(aveFit));

% Prepare table with headers
header = {'LearningRate', 'Neurons', 'Momentum', 'Fitness', 'RMSE', 'MAE', 'R2', 'TrainingTime_sec'};
T = array2table(results, 'VariableNames', header);
T.Trial = (1:iters)';  % Add trial number
T = movevars(T, 'Trial', 'Before', 1); % Make Trial first column

% Add final stats as new rows (with NaN placeholders)
stats = {
    'Best', NaN, NaN, NaN, min(aveFit), NaN, NaN, NaN, NaN;
    'Worst', NaN, NaN, NaN, max(aveFit), NaN, NaN, NaN, NaN;
    'Median', NaN, NaN, NaN, median(aveFit), NaN, NaN, NaN, NaN;
    'Mean', NaN, NaN, NaN, mean(aveFit), NaN, NaN, NaN, NaN;
    'StDev', NaN, NaN, NaN, std(aveFit), NaN, NaN, NaN, NaN;
};

statsTable = cell2table(stats, 'VariableNames', T.Properties.VariableNames);

% Ensure 'Trial' is a cell array of strings in both tables
if ~iscell(T.Trial)
    T.Trial = cellstr(string(T.Trial));
end

if ~iscell(statsTable.Trial)
    statsTable.Trial = cellstr(string(statsTable.Trial));
end

% Combine and save
finalTable = [T; statsTable];
writetable(finalTable, 'final_results.csv');

disp('Results saved to final_results.csv');
