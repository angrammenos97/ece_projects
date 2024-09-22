function curProc = sjf(arrival, flagPreempt)

%% DECLARE PERSISTENT VALUES
persistent pQueueNames
persistent pQueueTimes

%% INITIALIZATION

%% PARSE INPUT
if (not(isempty(pQueueNames)))
    pQueueTimes(1) = pQueueTimes(1) - 1; % one timestep passed
    if (pQueueTimes(1) <= 0) % job done so remove it
        pQueueNames = pQueueNames(2:end);
        pQueueTimes = pQueueTimes(2:end);
    end
end

if (not(isempty(arrival))) % add new process to queue
    if isempty(pQueueNames) % queue is empty
        pQueueNames = arrival(1,1);
        pQueueTimes = cell2mat(arrival(1,2));
    else % find where to add new process
        qStart = 2;
        if flagPreempt % policy is preemptive
            qStart = 1;
        end
        indexQ = size(pQueueTimes, 2) + 1; % index to insert new job
        for i = qStart:size(pQueueTimes, 2)
            if (cell2mat(arrival(1,2)) < pQueueTimes(i))
                indexQ = i;
                break
            end
        end        
        pQueueNames(indexQ+1:end+1) = pQueueNames(indexQ:end); % insert job name into queue
        pQueueNames(indexQ) = arrival(1,1);
        pQueueTimes = [pQueueTimes(1:indexQ-1), cell2mat(arrival(1,2)), pQueueTimes(indexQ:end)]; % and it's time
    end
end

if (not(isempty(pQueueNames)))
    curProc = pQueueNames{1}; % return current running job
else
    curProc = '_'; % no new job and empty queue
end

end