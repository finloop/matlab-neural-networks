clear;
load("OptDigits5.mat")
X = OptDigits5(:,1:end-1); 
Y = OptDigits5(:,end);

classes = unique(Y);
i = 1;
RES = table();
for class = classes'
    ind_1 = find(Y == class);
    X1 = X(ind_1,:);
    k1 = round(0.4 * length(ind_1)); % 40 % klas
    [k_1,Klastry_1] = kmeans(X1, k1, 'Distance', 'sqeuclidean');
    RES = [RES; array2table([Klastry_1 i*ones(k1, 1)])]; 
    i = i + 1;
end

XY = table2array(RES);
save('XYNEW.mat')
% Dla nowych danych odpalic najlepszy algorytm

