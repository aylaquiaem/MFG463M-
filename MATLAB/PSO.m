function [bestsol, bestfit, hist] = PSO(func, dim, bounds, steps, size, w, c1, c2, seed, vis)
% Pure PSO Algorithm

%% Init global
rng(seed);
bestsol = zeros(1, dim);
bestfit = inf;
lb = bounds(:, 1)';
ub = bounds(:, 2)';
hist = zeros(1, steps);

%% Init population and fitness
pop = lb + rand(size, dim) .* (ub - lb);
fit = arrayfun(@(i) func(pop(i, :)), 1:size);
range = (ub-lb) * 0.5;
vel = (2 * rand(size, dim) - 1) .* range;
sbpop = pop;
sbfit = fit;

%% Init 2D Visualization
if dim == 2 && vis == true
    figure;
    hold on;
    x = linspace(lb(1), ub(1), 100);
    y = linspace(lb(2), ub(2), 100);
    [X, Y] = meshgrid(x, y);
    Z = arrayfun(@(a, b) func([a, b]), X, Y);
    contour(X, Y, Z, 30, 'LineWidth', 1.5);
    colormap(turbo);
    colorbar;

    xlim([lb(1), ub(1)]);
    ylim([lb(2), ub(2)]);
    xlabel('x'); ylabel('y');
    grid on; 
    axis square;
    
    h = scatter(pop(:,1), pop(:,2), 20, 'r', 'filled');
    h_best = scatter(bestsol(1), bestsol(2), 40, 'k', 'filled');
end

%% Main Loop
for t = 1:steps
    % Update bests
    for i = 1:size
        fit(i) = func(pop(i,:));

        if fit(i) < sbfit(i)
            sbpop(i,:) = pop(i,:);
            sbfit(i) = fit(i);
        end

        if fit(i) < bestfit
            bestsol = pop(i,:);
            bestfit = fit(i);
        end
    end

    hist(t) = bestfit;

    % PSO update
    for i = 1:size
        vel(i,:) = w * vel(i,:) ...
                 + c1 * rand() * (sbpop(i,:) - pop(i,:)) ...
                 + c2 * rand() * (bestsol - pop(i,:));

        pop(i,:) = pop(i,:) + vel(i,:);

        % Bound handling
        pop(i,:) = max(min(pop(i,:), ub), lb);
    end

    % 2D Visualization
    if dim == 2 && vis == true
        set(h, 'XData', pop(:,1), 'YData', pop(:,2));
        set(h_best, 'XData', bestsol(1), 'YData', bestsol(2));
        title(sprintf('PSO Step %d | Best: %.12f', t, bestfit));
        drawnow;
    end
end
end
