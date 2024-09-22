% search the tree
%clear;
%data;
q = ceil(max(X, [], 'all')) * rand(1,dim); % random query point (q)
k = 3;
tau = inf; % desired search radius
list = repmat([0 tau], k, 1); % list with all neighbors
[list, tau] = searchNb(tree, list, tau, q);
% plot them to confirm
clf; hold on; axis equal
Visualize(X, list, q, dim);

function [list, tau] = searchNb(tree, list, tau, q)
    if (isempty(tree) ) % return if current vp is a leaf
        return;
    else
        dist = sqrt( sum((tree.vp(1,:) - q(1,:)).^2,2) ); % distance q of current vp        
        if (dist < tau) % add current vp if it is close to q
            tauId = find(list(:,2) == tau); % replace the farest point of the list with the current vp
            list(tauId(1),:) = [tree.idx, dist];
            M = max(list); % store the distance of the farest point of the list
            tau = M(1,2);
        end
        if (dist < tree.md) % q inside of the vp's circle
            [list, tau] = searchNb(tree.inner, list, tau,q);
            if (tree.md < (dist + tau))
                [list, tau] = searchNb(tree.outer, list, tau, q);
            end
        else % q outside of the vp's circle
            [list, tau] = searchNb(tree.outer, list, tau,q);
            if ((tree.md + tau)> dist)
                [list, tau] = searchNb(tree.inner, list, tau, q);
            end
        end       
    end    
end

function Visualize(X, list, q, dim)
    if (dim == 1)
        viscircles([q(:,1), 1], max(list(:,2)), 'Color', 'y');
        plot(X, 1, 'b.')
        plot(X(list(:,1)), 1, 'r.')
        plot(q(:,1), 1, 'go')
    elseif (dim == 2)
        viscircles(q, max(list(:,2)), 'Color', 'y', 'LineWidth', 0.1);
        plot(X(:,1), X(:,2), 'b.')
        plot(X(list(:,1),1), X(list(:,1),2), 'r.')
        plot(q(:,1), q(:,2), 'go')
    elseif (dim == 3)
        plot3(X(:,1), X(:,2), X(:,3), 'b.')
        plot3(X(list(:,1),1), X(list(:,1),2), X(list(:,1),3), 'r.')
        plot3(q(:,1), q(:,2), q(:,3), 'go')
    end
end