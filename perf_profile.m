%     This subroutine produces a data profile as described in:
%
%     Benchmarking Derivative-Free Optimization Algorithms
%     Jorge J. More' and Stefan M. Wild
%     SIAM J. Optimization, Vol. 20 (1), pp.172-191, 2009.

function hl = perf_profile(H,solvers,figtitle)

[np,ns] = size(H); % Grab the dimensions


% For each problem and solver, determine the number of evaluations
% required to reach the cutoff value
T = H;

% Other colors, lines, and markers are easily possible:
colors  = ['b' 'r' 'k' 'm' 'c' 'g' 'y'];   
lines   = {'-' '-.' '--'};
markers = ['o' 's' '^' 'p' '+' '*' 'x' 'v' '>' '<' 'd' 'h'];
%colors  = ['k' 'm' 'c' 'g'];
%colors  = ['b' 'r' 'g'];
%lines   = {'-' '--' ':' '-.'};
%markers = ['p' 'o' '^' 's'];

% Compute ratios and divide by smallest element in each row.
r = T./repmat(min(T,[],2),1,ns);

% Replace all NaN's with twice the max_ratio and sort.
max_ratio = max(max(r));
%max_ratio = 40 
%disp('perf_profile: max_ratio impostato a mano')
r(isnan(r)) = 2*max_ratio;
r = sort(r);

%ax(1) = axes('Position',[0.1 0.1 0.5 0.8]);
% Plot stair graphs with markers.
hl = zeros(ns,1);
for s = 1:ns
    [xs,ys] = stairs(r(:,s),(1:np)/np);

    % Only plot one marker at the intercept
    if (xs(1)==1)
        vv = find(xs==1,1,'last');
        xs = xs(vv:end);   ys = ys(vv:end);
    end

    sl = mod(s-1,3) + 1; sc = mod(s-1,7) + 1; sm = mod(s-1,12) + 1;
    option1 = [char(lines(sl)) colors(sc) markers(sm)];
    %hl(s) = semilogx(xs,ys,option1);
    hl(s) = plot(xs,ys,option1);
    hold on;
end

% Axis properties are set so that failures are not shown, but with the
% max_ratio data points shown. This highlights the "flatline" effect.
%axis([1 5 0 1]);
%axis([1 100 0 1]);
axis([1 max_ratio 0 1]);
legend(solvers,'Location','SouthEast');
xlabel('performance ratio, \alpha');
ylabel('percentage of problems');
title(figtitle);

xmed = 5;

if 0
    ax(2) = axes('Position',[0.7 0.1 0.2 0.8]);
    for s = 1:ns
        [xs,ys] = stairs(r(:,s),(1:np)/np);

        % Only plot one marker at the intercept
        %if (xs(1)==5)
            vv = find(xs>=xmed,1,'first');
            xs = xs(vv:end);   ys = ys(vv:end);
        %end

        sl = mod(s-1,3) + 1; sc = mod(s-1,7) + 1; sm = mod(s-1,12) + 1;
        option1 = [char(lines(sl)) colors(sc) markers(sm)];
        hl(s) = plot(xs,ys,option1);
        hold on;
    end
    axis([xmed max_ratio 0 1]);
    %ax(2).XScale = 'log';
    set(ax(2),'xtickmode','auto');
    ax(2).YTickLabel = {};
end

hold off;


end