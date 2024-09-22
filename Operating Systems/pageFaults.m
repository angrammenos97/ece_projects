function status = pageFaults(pageReq, nFrame)
    persistent frames
    
    status = 'M'; % default status is miss
    for i = 1:size(frames, 1) % add plus one to each page counter
        frames(i, 2) = frames(i, 2) + 1;
    end    
    if size(frames, 1) < nFrame % check if the set can hold more pages 
        if (find(frames == pageReq)) && (find(frames(:,1) == pageReq))
            frames(frames(:,1) == pageReq, 2) = 0; % if present reset counter
            status = 'H'; % change to hit
        else % insert if doesn t exist
            frames = [frames; pageReq 0];            
        end
    else % if memory is already full
        if find(frames(:,1)== pageReq) % if present reset counter
            frames(frames(:,1) == pageReq, 2) = 0;
            status = 'H'; % change to hit
        else % find page with the maximun couter value and replace it 
            lru = -1; % variable to hold max size
            val = 0; % index of the apove variable
            for i = 1:size(frames, 1) % find max counter
                if frames(i, 2) > lru
                    lru = frames(i, 2);
                    val = i;
                end
            end
            frames(val, :) = [pageReq 0]; % replace with the new
        end
    end
    frames
end