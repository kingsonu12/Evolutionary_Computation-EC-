function solve_large_gap()
    % Iterate through gap1 to gap12
    for g = 1:12
        filename = sprintf('/MATLAB Drive/Matlab Assignment/Assignment 4/gap%d.txt', g);
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
            
            % Solve the problem using Differential Evolution (DE)
            x_matrix = solve_gap_de(m, n, c, r, b);
            objective_value = sum(sum(c .* x_matrix));
            
            % Print formatted output
            fprintf('c%d-%d  %d\n', m*100 + n, p, round(objective_value));
        end
        
        % Close file
        fclose(fid);
    end
end

function x_matrix = solve_gap_de(m, n, c, r, b)
    % Parameters for Differential Evolution
    pop_size = 100; % Population size
    max_iter = 300; % Maximum iterations
    F = 0.5;        % Differential weight
    CR = 0.9;       % Crossover probability
    
    % Initialize population randomly (binary decision variables)
    pop = round(rand(pop_size, m * n));
    
    % Evaluate initial fitness
    fitness = arrayfun(@(i) fitnessFcn(pop(i, :)), 1:pop_size);
    
    % DE Main Loop
    for iter = 1:max_iter
        for i = 1:pop_size
            % Mutation
            idxs = randperm(pop_size, 3);
            while any(idxs == i)
                idxs = randperm(pop_size, 3);
            end
            v = pop(idxs(1), :) + F * (pop(idxs(2), :) - pop(idxs(3), :));
            v = round(1 ./ (1 + exp(-v))); % Sigmoid function for binary conversion
            
            % Crossover
            j_rand = randi(m * n);
            u = pop(i, :);
            for j = 1:m * n
                if rand < CR || j == j_rand
                    u(j) = v(j);
                end
            end
            
            % Ensure feasibility
            u = enforce_feasibility(u, m, n);
            
            % Selection
            u_fitness = fitnessFcn(u);
            if u_fitness < fitness(i)
                pop(i, :) = u;
                fitness(i) = u_fitness;
            end
        end
    end
    
    % Get best solution
    [~, best_idx] = min(fitness);
    x_matrix = reshape(pop(best_idx, :), [m, n]);
    
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
