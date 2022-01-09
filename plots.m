%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear, clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lettura dei valori iniziali
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen('initial_value.txt','r'); %column f_0 of the initial function value
F_ZERO = textscan(fid,'%s&%d&%d&%f','Headerlines',1);
F_ZERO = F_ZERO{1,4};
fclose(fid);

i = 1;
RES = {};

A.descr = 'Non-Mon_Dembo_Curv.xlsx';
A.label = 'Dembo curv. (NM-M)';
A.id = i;
temp = xlsread(A.descr);
A.f = temp(:,7);
A.gnr = temp(:,8);
A.cpu = temp(:,9);
A.iter = temp(:,3);
A.nf = temp(:,4);
A.ng = temp(:,5);
RES{i} = A;
i = i+1;

A.descr = 'Non-Mon_Dembo_Nocurv.xlsx';
A.label = 'Dembo no curv. (NM-M)';
A.id = i;
temp = xlsread(A.descr);
A.f = temp(:,7);
A.gnr = temp(:,8);
A.cpu = temp(:,9);
A.iter = temp(:,3);
A.nf = temp(:,4);
A.ng = temp(:,5);
RES{i} = A;
i = i+1;

A.descr = 'Non-Mon_Nash_Curv.xlsx';
A.label = 'Nash curv. (NM-M)';
A.id = i;
temp = xlsread(A.descr);
A.f = temp(:,7);
A.gnr = temp(:,8);
A.cpu = temp(:,9);
A.iter = temp(:,3);
A.nf = temp(:,4);
A.ng = temp(:,5);
RES{i} = A;
i = i+1;

A.descr = 'Non-Mon_Nash_Nocurv.xlsx';
A.label = 'Nash no curv. (NM-M)';
A.id = i;
temp = xlsread(A.descr);
A.f = temp(:,7);
A.gnr = temp(:,8);
A.cpu = temp(:,9);
A.iter = temp(:,3);
A.nf = temp(:,4);
A.ng = temp(:,5);
RES{i} = A;
i = i+1;

A.descr = 'Mon_Dembo_Curv.xlsx';
A.label = 'Dembo curv. (M)';
A.id = i;
temp = xlsread(A.descr);
A.f = temp(:,7);
A.gnr = temp(:,8);
A.cpu = temp(:,9);
A.iter = temp(:,3);
A.nf = temp(:,4);
A.ng = temp(:,5);
RES{i} = A;
i = i+1;

A.descr = 'Mon_Dembo_Nocurv.xlsx';
A.label = 'Dembo no curv. (M)';
A.id = i;
temp = xlsread(A.descr);
A.f = temp(:,7);
A.gnr = temp(:,8);
A.cpu = temp(:,9);
A.iter = temp(:,3);
A.nf = temp(:,4);
A.ng = temp(:,5);
RES{i} = A;
i = i+1;

A.descr = 'Mon_Nash_Curv.xlsx';
A.label = 'Nash curv. (M)';
A.id = i;
temp = xlsread(A.descr);
A.f = temp(:,7);
A.gnr = temp(:,8);
A.cpu = temp(:,9);
A.iter = temp(:,3);
A.nf = temp(:,4);
A.ng = temp(:,5);
RES{i} = A;
i = i+1;

A.descr = 'Mon_Nash_Nocurv.xlsx';
A.label = 'Nash no curv. (M)';
A.id = i;
temp = xlsread(A.descr);
A.f = temp(:,7);
A.gnr = temp(:,8);
A.cpu = temp(:,9);
A.iter = temp(:,3);
A.nf = temp(:,4);
A.ng = temp(:,5);
RES{i} = A;
i = i+1;

A.descr = 'Non_Dembo_Curv.xlsx';
A.label = 'Dembo curv. (NM)';
A.id = i;
temp = xlsread(A.descr);
A.f = temp(:,7);
A.gnr = temp(:,8);
A.cpu = temp(:,9);
A.iter = temp(:,3);
A.nf = temp(:,4);
A.ng = temp(:,5);
RES{i} = A;
i = i+1;

A.descr = 'Non_Dembo_Nocurv.xlsx';
A.label = 'Dembo no curv. (NM)';
A.id = i;
temp = xlsread(A.descr);
A.f = temp(:,7);
A.gnr = temp(:,8);
A.cpu = temp(:,9);
A.iter = temp(:,3);
A.nf = temp(:,4);
A.ng = temp(:,5);
RES{i} = A;
i = i+1;

A.descr = 'Non_Nash_Curv.xlsx';
A.label = 'Nash curv. (NM)';
A.id = i;
temp = xlsread(A.descr);
A.f = temp(:,7);
A.gnr = temp(:,8);
A.cpu = temp(:,9);
A.iter = temp(:,3);
A.nf = temp(:,4);
A.ng = temp(:,5);
RES{i} = A;
i = i+1;

A.descr = 'Non_Nash_Nocurv.xlsx';
A.label = 'Nash no curv. (NM)';
A.id = i;
temp = xlsread(A.descr);
A.f = temp(:,7);
A.gnr = temp(:,8);
A.cpu = temp(:,9);
A.iter = temp(:,3);
A.nf = temp(:,4);
A.ng = temp(:,5);
RES{i} = A;
i = i+1;

[temp, nres] = size(RES);

fprintf('\n\n');
fprintf('Available results:\n');
for i = 1:nres
    fprintf("%3d %30s\n",RES{i}.id,RES{i}.label);
end
SEL = [];
while true
    fprintf('Select the id of a result (%d-%d) or x to exit: ',1,nres);
    choice = input('','s');
    if strcmp(choice,'x') || strcmp(choice,'')
        if length(SEL) > 1
            break
        else
            fprintf('è necessario selezionare almeno due algoritmi\n');
            continue
        end
    end
    id = str2num(choice);
    SEL = [SEL, id];
    % fprintf('%d\n',id)
end

solvers = {}; 
for i = 1:length(SEL)
    solvers(i) = {RES{SEL(i)}.label};
end
disp(solvers);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% selection of the performance index
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Available performance indices:\n');
fprintf('1 - CPU time\n');
fprintf('2 - function evaluations\n');
fprintf('3 - gradient evaluations\n');
fprintf('4 - outer iterations\n');
while true
    fprintf('Select the id of one performance index (%d-%d): ',1,4);
    choice = input('','s');
    if strcmp(choice,'')
        fprintf('è necessario selezionare un indice\n');
        continue
    end
    perfid = str2num(choice);
    break
end

ns = length(solvers);
[np,~] = size(RES{SEL(1)}.f);

H = zeros(np,ns);
T = zeros(np,ns);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% indice di prestazione: TEMPO (posizione 11)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tau = 1.e-4;

for i = 1:length(SEL)
    T(:,i) = RES{SEL(i)}.f;
    if perfid == 1
        H(:,i) = RES{SEL(i)}.cpu;
    elseif perfid == 2
        H(:,i) = RES{SEL(i)}.nf;
    elseif perfid == 3
        H(:,i) = RES{SEL(i)}.ng;
    else
        H(:,i) = RES{SEL(i)}.iter;
    end
end
F_BEST = min(T,[],2);
infvalue = 2*max(max(H));

for i = 1:length(SEL)
    fall = F_ZERO - T(:,i) < (1-tau)*(F_ZERO-F_BEST);
    H(fall,i) = infvalue;
    %H(fallimenti2,2) = infvalue;
    %fallimenti2 = F_ZERO - T(:,2) < (1-tau)*(F_ZERO-F_BEST);
end

if perfid == 1
    figtitle = 'CPU time';
elseif perfid == 2
    figtitle = 'function evaluations';
elseif perfid == 3
    figtitle = 'gradient evaluations';
else
    figtitle = 'outer iterations';
end

figure('Position',[0,0,1200,1000])
perf_profile(H,solvers,figtitle);
