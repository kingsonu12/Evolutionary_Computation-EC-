function run_gap_ga_solver()
    % Iterate through gap1 to gap12 dataset files
    for file_idx = 1:12
        file_name = sprintf('gap%d.txt', file_idx);
        file_id = fopen(file_name, 'r');
        if file_id == -1
            error('Error opening file %s.', file_name);
        end
        
        % Read the number of problem sets
        num_problems = fscanf(file_id, '%d', 1);
        
        % Print dataset name (gapX)
        fprintf('\n%s\n', file_name(1:end-4)); % Removes .txt for display
        
        for problem_idx = 1:num_problems
            % Read problem parameters
            num_servers = fscanf(file_id, '%d', 1); % Number of servers
            num_users = fscanf(file_id, '%d', 1); % Number of users
            
            % Read cost and resource matrices
            cost_matrix = fscanf(file_id, '%d', [num_users, num_servers])';
            resource_matrix = fscanf(file_id, '%d', [num_users, num_servers])';
            
            % Read server capacities
            server_capacities = fscanf(file_id, '%d', [num_servers, 1]);
            
            % Solve using Genetic Algorithm (GA)
            assignment_matrix = solve_ga_assignment(num_servers, num_users, cost_matrix, resource_matrix, server_capacities);
            objective_value = sum(sum(cost_matrix .* assignment_matrix)); % Maximization
            
            % Print formatted output
            fprintf('c%d-%d  %d\n', num_servers*100 + num_users, problem_idx, round(objective_value));
        end
        
        % Close file
        fclose(file_id);
    end
end

function assignment_matrix = solve_ga_assignment(num_servers, num_users, cost_matrix, resource_matrix, server_capacities)
    % GA Parameters
    population_size = 100; % Population size
    max_generations = 300;  % Maximum generations
    crossover_probability = 0.8;
    mutation_probability = 0.02;
    
    % Initialize population randomly
    population = round(rand(population_size, num_servers * num_users));

    % Make initial solution feasible
    for i = 1:population_size
        population(i, :) = enforce_solution_feasibility(rand(1, num_servers * num_users), num_servers, num_users);
    end
    
    % Evaluate initial fitness
    fitness_values = arrayfun(@(i) evaluate_fitness(population(i, :)), 1:population_size);
    
    % Main GA loop
    for generation = 1:max_generations
        % Selection (Tournament Selection)
        selected_parents = tournament_selection(population, fitness_values);
        
        % Crossover (Single Point)
        offspring_population = single_point_crossover(selected_parents, crossover_probability);
        
        % Mutation (Random Flip)
        mutated_offspring = mutation(offspring_population, mutation_probability);
        
        % Ensure feasibility
        for i = 1:size(mutated_offspring, 1)
            mutated_offspring(i, :) = enforce_solution_feasibility(mutated_offspring(i, :), num_servers, num_users);
        end
        
        % Evaluate fitness of the mutated offspring
        new_fitness_values = arrayfun(@(i) evaluate_fitness(mutated_offspring(i, :)), 1:size(mutated_offspring, 1));
        
        % Elitism (Keep the best)
        [~, best_idx] = max([fitness_values, new_fitness_values]); % Maximization
        if best_idx > length(fitness_values)
            population = mutated_offspring;
            fitness_values = new_fitness_values;
        else
            population = [population; mutated_offspring];
            fitness_values = [fitness_values, new_fitness_values];
        end
        
        % Select top individuals
        [~, sorted_idx] = sort(fitness_values, 'descend'); % Maximization
        population = population(sorted_idx(1:population_size), :);
        fitness_values = fitness_values(sorted_idx(1:population_size));
    end
    
    % Return the best solution
    [~, best_idx] = max(fitness_values);
    assignment_matrix = reshape(population(best_idx, :), [num_servers, num_users]);

    function fitness_value = evaluate_fitness(solution)
        solution_matrix = reshape(solution, [num_servers, num_users]);
        total_cost = sum(sum(cost_matrix .* solution_matrix)); % Maximization
        
        % Constraint violations (penalty approach)
        capacity_violation = sum(max(sum(solution_matrix .* resource_matrix, 2) - server_capacities, 0)); % Server capacity
        assignment_violation = sum(abs(sum(solution_matrix, 1) - 1)); % Each user assigned exactly once
        penalty_term = 1e6 * (capacity_violation + assignment_violation);
        
        fitness_value = total_cost - penalty_term; % Maximization
    end
end

function selected_parents = tournament_selection(population, fitness_values)
    % Tournament selection
    population_size = size(population, 1);
    selected_parents = zeros(size(population));
    
    for i = 1:population_size
        % Randomly pick two individuals
        idx1 = randi(population_size);
        idx2 = randi(population_size);
        
        % Select the better one
        if fitness_values(idx1) > fitness_values(idx2) % Maximization
            selected_parents(i, :) = population(idx1, :);
        else
            selected_parents(i, :) = population(idx2, :);
        end
    end
end

function offspring_population = single_point_crossover(parents, crossover_probability)
    population_size = size(parents, 1);
    num_genes = size(parents, 2);
    offspring_population = parents;
    
    for i = 1:2:population_size-1
        if rand < crossover_probability
            % Choose a random crossover point
            crossover_point = randi(num_genes - 1);
            offspring_population(i, crossover_point+1:end) = parents(i+1, crossover_point+1:end);
            offspring_population(i+1, crossover_point+1:end) = parents(i, crossover_point+1:end);
        end
    end
end

function mutated_population = mutation(offspring_population, mutation_probability)
    mutated_population = offspring_population;
    for i = 1:numel(offspring_population)
        if rand < mutation_probability
            mutated_population(i) = 1 - mutated_population(i); % Flip bit
        end
    end
end

function feasible_solution = enforce_solution_feasibility(solution, num_servers, num_users)
    % Ensure each user is assigned exactly once
    solution_matrix = reshape(solution, [num_servers, num_users]);
    for user_idx = 1:num_users
        [~, best_server_idx] = max(solution_matrix(:, user_idx)); % Assign to the best server
        solution_matrix(:, user_idx) = 0;
        solution_matrix(best_server_idx, user_idx) = 1;
    end
    feasible_solution = reshape(solution_matrix, [1, num_servers * num_users]);
end