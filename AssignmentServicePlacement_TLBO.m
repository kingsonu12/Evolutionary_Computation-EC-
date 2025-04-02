function solve_large_gap_TLBO()
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
            
            % Solve the problem using TLBO
            x_matrix = solve_gap_TLBO(m, n, c, r, b);
            objective_value = sum(sum(c .* x_matrix));
            
            % Print formatted output
            fprintf('c%d-%d  %d\n', m*100 + n, p, round(objective_value));
        end
        
        % Close file
        fclose(fid);
    end
end

function x_matrix = solve_gap_TLBO(m, n, c, r, b)
    % Parameters for TLBO
    num_learners = 100;  % Population size
    max_iter = 10;      % Maximum iterations
    
    % Initialize learners randomly (binary decision variables)
    learners = round(rand(num_learners, m * n));
    
    % Evaluate initial fitness
    fitness = arrayfun(@(i) fitnessFcn(learners(i, :)), 1:num_learners);
    
    % TLBO Main Loop
    for iter = 1:max_iter
        % Teacher Phase
        [~, teacher_idx] = min(fitness);
        teacher = learners(teacher_idx, :);
        
        for i = 1:num_learners
            TF = randi([1, 2]); % Teaching Factor
            learners(i, :) = learners(i, :) + rand * (teacher - TF * mean(learners));
            
            % Sigmoid-based binary conversion
            learners(i, :) = round(1 ./ (1 + exp(-learners(i, :))));
            
            % Ensure feasibility
            learners(i, :) = enforce_feasibility(learners(i, :), m, n);
        end
        
        % Learner Phase
        for i = 1:num_learners
            partner_idx = randi([1, num_learners]);
            if fitness(i) < fitness(partner_idx)
                learners(i, :) = learners(i, :) + rand * (learners(i, :) - learners(partner_idx, :));
            else
                learners(i, :) = learners(i, :) + rand * (learners(partner_idx, :) - learners(i, :));
            end
            
            % Sigmoid-based binary conversion
            learners(i, :) = round(1 ./ (1 + exp(-learners(i, :))));
            
            % Ensure feasibility
            learners(i, :) = enforce_feasibility(learners(i, :), m, n);
        end
        
        % Evaluate fitness
        fitness = arrayfun(@(i) fitnessFcn(learners(i, :)), 1:num_learners);
    end
    
    % Reshape the best solution found
    [~, best_idx] = min(fitness);
    x_matrix = reshape(learners(best_idx, :), [m, n]);

    function fval = fitnessFcn(x)
        x_mat = reshape(x, [m, n]);
        cost = sum(sum(c .* x_mat));
        
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
