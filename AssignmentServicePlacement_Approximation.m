function finalFitnessMatrices = processAllGapInstances()
    numFiles = 12; % Number of gap files
    finalFitnessMatrices = cell(1, numFiles); % Store final fitness matrices
    outputData = cell(numFiles, 1); % Store output strings in rows
    
    for fileIdx = 1:numFiles
        % Construct filename
        fileName = sprintf('gap%d.txt', fileIdx);
        
        % Open file
        fileID = fopen(fileName, 'r');
        if fileID == -1
            error('Error opening file %s.', fileName);
        end
        
        % Read number of problem instances
        numInstances = fscanf(fileID, '%d', 1);
        fileOutput = cell(numInstances, 1);
        
        for instanceIdx = 1:numInstances
            % Read the number of machines and tasks
            numMachines = fscanf(fileID, '%d', 1); 
            numTasks = fscanf(fileID, '%d', 1); 
            
            % Read profit matrix
            profitMatrix = fscanf(fileID, '%d', [numTasks, numMachines])';
            
            % Read resource demand matrix
            demandMatrix = fscanf(fileID, '%d', [numTasks, numMachines])';
            
            % Read machine capacity constraints
            capacityVector = fscanf(fileID, '%d', [numMachines, 1]);
            
            % Solve the GAP using a heuristic allocation strategy
            assignmentMatrix = heuristicGapSolver(numMachines, numTasks, profitMatrix, demandMatrix, capacityVector);
            
            % Compute total assigned profit
            totalProfit = sum(sum(profitMatrix .* assignmentMatrix));
            
            % Store the final fitness matrix
            finalFitnessMatrices{fileIdx} = assignmentMatrix;
            
            % Store formatted output for aligned printing
            fileOutput{instanceIdx} = sprintf('c%d%d-%d %d', numMachines, numTasks, instanceIdx, totalProfit);
        end
        outputData{fileIdx} = fileOutput;
        
        % Close file
        fclose(fileID);
    end
    
    % Determine the max number of instances
    maxInstances = max(cellfun(@length, outputData));
    
    % Print outputs side by side, 4 files per row
    for fileStart = 1:4:numFiles
        % Print headers
        for fileIdx = fileStart:min(fileStart+3, numFiles)
            fprintf('gap%d\t\t', fileIdx);
        end
        fprintf('\n');
        
        % Print aligned results
        for instanceIdx = 1:maxInstances
            for fileIdx = fileStart:min(fileStart+3, numFiles)
                if instanceIdx <= length(outputData{fileIdx})
                    fprintf('%s\t', outputData{fileIdx}{instanceIdx});
                else
                    fprintf('\t');
                end
            end
            fprintf('\n');
        end
        fprintf('\n');
    end
end

function allocationMatrix = heuristicGapSolver(numMachines, numTasks, profitMatrix, demandMatrix, capacityVector)
    % Initialize allocation matrix
    allocationMatrix = zeros(numMachines, numTasks);
    
    % Compute benefit-to-demand ratio
    efficiencyScore = -profitMatrix ./ (demandMatrix + 1e-6); % Avoid division by zero
    
    % Flatten indices and sort tasks by highest efficiency
    [~, sortedTaskIndices] = sort(efficiencyScore(:), 'descend');
    
    % Track remaining capacities
    remainingCapacity = capacityVector;
    
    for idx = sortedTaskIndices'
        % Determine the (machine, task) pair
        [machineIdx, taskIdx] = ind2sub([numMachines, numTasks], idx);
        
        % Assign task to machine if capacity allows
        if remainingCapacity(machineIdx) >= demandMatrix(machineIdx, taskIdx)
            allocationMatrix(machineIdx, taskIdx) = 1;
            remainingCapacity(machineIdx) = remainingCapacity(machineIdx) - demandMatrix(machineIdx, taskIdx);
        end
    end
end