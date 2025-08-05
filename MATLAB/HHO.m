function [bestsol, bestfit, hist] = HHO(func, dim, bounds, steps, size, seed, vis)
% Pure HHO 

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
    
    h = scatter(pop(:,1), pop(:,2), 20, 'r', 'filled'); % agents
    h_best = scatter(bestsol(1), bestsol(2), 40, 'k', 'filled'); % best
end

%% Main Loop
for t = 1:steps
    % Update global best
    for i = 1:size
        if fit(i) < bestfit
            bestfit = fit(i);
            bestsol = pop(i,:);
        end
    end

    hist(t) = bestfit;
    popM = mean(pop);

    % Main HHO update
    for i = 1:size
        E1 = 2 * (1 - t/steps);
        E0 = 2 * rand() - 1;
        E = E1 * E0;
        r = rand();
        J = 2 * (1 - rand());

        if abs(E) >= 1
            % Exploration phase
            q = rand();
            popR = pop(randi(size), :);
            if q >= 0.5
                pop(i,:) = popR - rand() * abs(popR - 2 * rand() * pop(i,:));
            else
                pop(i,:) = (bestsol - popM) - rand() * ((ub - lb) .* rand(1, dim) + lb);
            end
        else
            % Exploitation phase
            if r >= 0.5 && abs(E) >= 0.5
                % Soft besiege
                pop(i,:) = (bestsol - pop(i,:)) - E * abs(J * bestsol - pop(i,:));
            elseif r >= 0.5 && abs(E) < 0.5
                % Hard besiege
                pop(i,:) = bestsol - E * abs(bestsol - pop(i,:));
            elseif r < 0.5 && abs(E) >= 0.5
                % Soft besiege with progressive rapid dives
                Y = bestsol - E * abs(J * bestsol - pop(i,:));
                Z = Y + rand(1, dim) .* LF(dim);
                if func(Y) < fit(i)
                    pop(i,:) = Y;
                elseif func(Z) < fit(i)
                    pop(i,:) = Z;
                end
            elseif r < 0.5 && abs(E) < 0.5
                % Hard besiege with progressive rapid dives
                Y = bestsol - E * abs(J * bestsol - popM);
                Z = Y + rand(1, dim) .* LF(dim);
                if func(Y) < fit(i)
                    pop(i,:) = Y;
                elseif func(Z) < fit(i)
                    pop(i,:) = Z;
                end
            end
        end

        % Clamp to bounds
        pop(i,:) = max(min(pop(i,:), ub), lb);
        fit(i) = func(pop(i,:));
    end

    % 2D Visualization
    if dim == 2 && vis == true
        set(h, 'XData', pop(:,1), 'YData', pop(:,2));
        set(h_best, 'XData', bestsol(1), 'YData', bestsol(2));
        title(sprintf('HHO Step %d | Best: %.12f', t, bestfit));
        drawnow;
    end
end
end

%% Lévy Flight Helper
function y = LF(d)
beta = 1.5;
num = gamma(1+beta) * sin(pi*beta/2);
den = gamma((1+beta)/2) * beta * (2^((beta-1)/2));
sigma = (num / den) ^ (1/beta);
u = randn(1,d) * sigma;
v = randn(1,d);
y = u ./ (abs(v).^(1 / beta));
end
