function processDataFiles()
    totalFiles = 12;
    aggregatedResults = cell(totalFiles, 1);

    % Iterate through data1 to data12
    for fileIndex = 1:totalFiles
        fileName = sprintf('gap%d.txt', fileIndex);
        fileId = fopen(fileName, 'r');
        if fileId == -1
            error('Error opening file %s.', fileName);
        end

        % Read the number of test cases
        totalCases = fscanf(fileId, '%d', 1);
        caseResults = cell(totalCases, 1);

        for caseIndex = 1:totalCases
            % Read input parameters
            serverCount = fscanf(fileId, '%d', 1);
            userCount = fscanf(fileId, '%d', 1);

            % Read cost and resource matrices
            costMatrix = fscanf(fileId, '%d', [userCount, serverCount])';
            resourceMatrix = fscanf(fileId, '%d', [userCount, serverCount])';

            % Read server capacities
            capacityVector = fscanf(fileId, '%d', [serverCount, 1]);

            % Solve the problem
            xMatrix = solveGapMax(serverCount, userCount, costMatrix, resourceMatrix, capacityVector);
            totalCost = sum(sum(costMatrix .* xMatrix));

            % Store formatted output
            caseResults{caseIndex} = sprintf('c%d-%d\t%d', serverCount*100 + userCount, caseIndex, round(totalCost));
        end

        % Close file
        fclose(fileId);
        aggregatedResults{fileIndex} = caseResults;
    end

    % Display results side by side
    columnsPerRow = 4;
    for rowStart = 1:columnsPerRow:totalFiles
        rowEnd = min(rowStart + columnsPerRow - 1, totalFiles);

        % Print headers
        for fileIndex = rowStart:rowEnd
            fprintf('gap%d\t\t', fileIndex);
        end
        fprintf('\n');

        % Determine max number of cases in this row
        maxCases = max(cellfun(@length, aggregatedResults(rowStart:rowEnd)));

        % Print data row-wise
        for caseIndex = 1:maxCases
            for fileIndex = rowStart:rowEnd
                if caseIndex <= length(aggregatedResults{fileIndex})
                    fprintf('%s\t', aggregatedResults{fileIndex}{caseIndex});
                else
                    fprintf('\t\t');
                end
            end
            fprintf('\n');
        end
        fprintf('\n');
    end
end

function xMatrix = solveGapMax(m, n, c, r, b)
    f = -c(:); % Convert to column vector for maximization

    % Constraint 1: Each user assigned exactly once
    AeqJobs = kron(eye(n), ones(1, m));
    beqJobs = ones(n, 1);

    % Constraint 2: Server resource constraints
    AineqAgents = zeros(m, m * n);
    for i = 1:m
        for j = 1:n
            AineqAgents(i, (j-1)*m + i) = r(i,j);
        end
    end
    bineqAgents = b;

    % Define variable bounds (binary decision variables)
    lb = zeros(m * n, 1);
    ub = ones(m * n, 1);
    intcon = 1:(m*n);

    % Solve using intlinprog
    options = optimoptions('intlinprog', 'Display', 'off');
    x = intlinprog(f, intcon, AineqAgents, bineqAgents, AeqJobs, beqJobs, lb, ub, options);

    % Reshape into m × n matrix
    xMatrix = reshape(x, [m, n]);
end

% Run the function

 