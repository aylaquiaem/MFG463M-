function [bestsol, bestfit, hist] = PSOHHO(func, dim, bounds, steps, size, w, c1, c2, seed, vis)
% PSO Exploration, HHO Exploitation

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
    
    h = scatter(pop(:,1), pop(:,2), 20, 'r', 'filled'); % duelists
    h_best = scatter(bestsol(1), bestsol(2), 40, 'k', 'filled'); % best solution
end

%% Main Loop
for t = 1:steps
    % Update best
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

   % Hawk update
    for i = 1:size
        %  Select rand and calculate mean
        popM = mean(pop);
        r5 = rand();

        Eo = 2 * rand() - 1;
        J = 2 * (1-r5);
        E = 2 * Eo * (1 - t/steps);

        if abs(E) >= 1
            % Exploration (PSO)
            vel(i,:) = w * vel(i,:) + c1 * rand() * (sbpop(i,:)-pop(i,:)) + c2 * rand() * (bestsol-pop(i,:));
            pop(i,:) = pop(i,:) + vel(i,:);
        else
            % Exploitation (HHO)
            r = rand();
            if abs(E) >= 0.5 && r >= 0.5
                delta = bestsol - pop(i,:);
                pop(i,:) = delta - (E * abs((J*bestsol) - pop(i,:)));
            elseif abs(E) < 0.5 && r >= 0.5
                delta = bestsol- pop(i,:);
                pop(i,:) = bestsol - (E * abs(delta));
            elseif abs(E) >= 0.5 && r < 0.5
                Y = bestsol - (E * abs((J*bestsol) - pop(i,:)));
                Z = Y + (rand(1, dim) .* LF(dim));
                if func(Y) < fit(i)
                    pop(i,:) = Y;
                elseif func(Z) < fit(i)
                    pop(i,:) = Z;
                end
            elseif abs(E) < 0.5 && r < 0.5
                Y = bestsol - (E * abs((J*bestsol) - popM));
                Z = Y + (rand(1, dim) .* LF(dim));
                if func(Y) < fit(i)
                    pop(i,:) = Y;
                elseif func(Z) < fit(i)
                    pop(i,:) = Z;
                end
            end
            pop(i,:) = max(min(pop(i,:), ub), lb);
        end
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

function y = LF(d)
beta =  1.5;
num = gamma(1+beta) * sin(pi*beta/2);
den = gamma((1+beta)/2) * beta * (2 ^ ((beta-1)/2));
sigma = (num / den) ^ (1/beta);
u = randn(1,d)*sigma;
v = randn(1,d);
y = u ./ (abs(v) .^ (1 / beta));
end