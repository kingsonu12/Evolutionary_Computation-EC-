% PSO
function solve_large_gap()
    % Iterate through gap1 to gap12
    for g = 1:12
        filename = sprintf('/MATLAB Drive/Matlab Assignment/Assignment 3/gap%d.txt', g);
        fid = fopen(filename, 'r');
        if fid == -1
            error('Error opening file %s.', filename);
        end
           
        % Read the number of problem sets
        num_problems = fscanf(fid, '%d', 1);
        
        % Print dataset name (gapX)
        fprintf('\n%s\n', filename(1:end-4)); % Removes .txt for display
        
        for p = 1:num_problems
            % Read problem parameters
            m = fscanf(fid, '%d', 1); % Number of servers
            n = fscanf(fid, '%d', 1); % Number of users
            
            % Read cost and resource matrices
            c = fscanf(fid, '%d', [n, m])';
            r = fscanf(fid, '%d', [n, m])';
            
            % Read server capacities
            b = fscanf(fid, '%d', [m, 1]);
            
            % Solve the problem using Particle Swarm Optimization (PSO)
            x_matrix = solve_gap_pso(m, n, c, r, b);
            objective_value = sum(sum(c .* x_matrix));
            
            % Print formatted output
            fprintf('c%d-%d  %d\n', m*100 + n, p, round(objective_value));
        end
        
        % Close file
        fclose(fid);
    end
end

function x_matrix = solve_gap_pso(m, n, c, r, b)
    % Parameters for PSO
    num_particles = 100; % Swarm size
    max_iter = 10;      % Maximum iterations
    w = 0.7;             % Inertia weight
    c1 = 1.5;            % Cognitive coefficient
    c2 = 1.5;            % Social coefficient
    
    % Initialize particles randomly (binary decision variables)
    swarm = round(rand(num_particles, m * n));
    velocity = zeros(num_particles, m * n);
    
    % Evaluate initial fitness
    fitness = arrayfun(@(i) fitnessFcn(swarm(i, :)), 1:num_particles);
    
    % Initialize best positions
    p_best = swarm;
    p_best_fitness = fitness;
    
    % Global best
    [g_best_fitness, g_best_idx] = min(p_best_fitness);
    g_best = p_best(g_best_idx, :);
    
    % PSO Main Loop
    for iter = 1:max_iter
        for i = 1:num_particles
            % Update velocity
            velocity(i, :) = w * velocity(i, :) ...
                            + c1 * rand(1, m * n) .* (p_best(i, :) - swarm(i, :)) ...
                            + c2 * rand(1, m * n) .* (g_best - swarm(i, :));
                        
            % Update position (sigmoid-based binary conversion)
            swarm(i, :) = round(1 ./ (1 + exp(-velocity(i, :))));
            
            % Ensure feasibility (each user assigned to exactly one server)
            swarm(i, :) = enforce_feasibility(swarm(i, :), m, n);
            
            % Evaluate fitness
            fitness(i) = fitnessFcn(swarm(i, :));
            
            % Update personal best
            if fitness(i) < p_best_fitness(i)
                p_best(i, :) = swarm(i, :);
                p_best_fitness(i) = fitness(i);
            end
        end
        
        % Update global best
        [new_g_best_fitness, g_best_idx] = min(p_best_fitness);
        if new_g_best_fitness < g_best_fitness
            g_best = p_best(g_best_idx, :);
            g_best_fitness = new_g_best_fitness;
        end
    end
    
    % Reshape the best solution found
    x_matrix = reshape(g_best, [m, n]);

    function fval = fitnessFcn(x)
        x_mat = reshape(x, [m, n]);
        cost = -sum(sum(c .* x_mat));
        
        % Constraint violations (penalty approach)
        capacity_violation = sum(max(sum(x_mat .* r, 2) - b, 0));
        assignment_violation = sum(abs(sum(x_mat, 1) - 1));
        penalty = 1e6 * (capacity_violation + assignment_violation);
        
        fval = cost + penalty;
    end

    function x_corrected = enforce_feasibility(x, m, n)
        x_mat = reshape(x, [m, n]);
        for j = 1:n
            [~, idx] = max(x_mat(:, j)); % Assign to the best candidate
            x_mat(:, j) = 0;
            x_mat(idx, j) = 1;
        end
        x_corrected = reshape(x_mat, [1, m * n]);
    end
end

