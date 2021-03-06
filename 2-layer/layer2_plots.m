%%%%%%%%%%%%%%%%%%% Wykres wszystkie algorytmy %%%%%%%%%%%%%%%%%%%%%%%%%%%% 
clear
clc
training_functions = ["trainscg" "trainrp" "traincgb"];
activation_functions = ["tansig" "logsig"];

% Wygenerowanie indeksów do CV
% Wymagane jest zadanie wartości dla K.
L1_NEURONS = [20 40 60 120];
L2_NEURONS = [20 40 60 80];

NET_SIZES = zeros(length(L1_NEURONS)*length(L2_NEURONS), 2);
labels = strings(length(L1_NEURONS)*length(L2_NEURONS),1);

for L1=1:length(L1_NEURONS)
    for L2=1:length(L2_NEURONS)
        labels((L1-1)*length(L1_NEURONS)+L2) = mat2str([L1_NEURONS(L1) L2_NEURONS(L2)]);
    end
end

hold off
i = 1;
cont = {};

MM = zeros(2, 1);
for tfun = training_functions
    for afun = activation_functions
        load(tfun +"_"+ afun);
        [M, I] = max(mean(ACC, 2));
        if MM(1) < M
            MM = [M I];
        end

        plot(10:10:160,mean(ACC, 2), 'LineWidth', 2);
        ylim([0.5 1]);
        set(gca, 'Xtick', 10:10:160);
        xticklabels(labels)
        cont{i} = tfun +"--" + afun;

        hold on
        i = i +1;
        clear("ACC")
    end
end

plot(MM(2)*10, MM(1), '.', 'MarkerSize', 20)
cont{length(cont)+1}= "Punkt z największą dokładnością";
title(["Wykres dokładności od ilości neuronów dla rożnych", ...
    "algorytmów uczących i funkcji aktywacji"])
xlabel("Ilość neuronów w warstwie ukrytej")
ylabel("Dokładność")
legend(cont, 'Location', 'southwest')
saveas(gcf, 'img/precision-all.png')
close(gcf)

%%%%%%%%%%%%% Wykresy dokładności per klasa dla algorytmów %%%%%%%%%%%%%%%%%%%
training_functions = ["trainscg" "trainrp" "traincgb"];
activation_functions = ["tansig" "logsig"];

i = 1;
cont = {};

for tfun = training_functions
    for afun = activation_functions
        load(tfun +"_"+ afun);
        plot(ACC, 'LineWidth', 2);
        set(gca, 'Xtick', 1:1:16);
        xticklabels(labels)
        title(["Dokładność dla każdej z klas dla kombinacji", ...
            (tfun +"--"+ afun)])
        ylim([0.5 1]);
        for j=[1:12]
            cont{j} = num2str(j);
        end
        
        leg = legend(cont,'Location', 'southeast');
        title(leg, "Klasy");
        leg.Title.Visible = 'on';
        xlabel("Ilość neuronów w warstwie ukrytej")
        ylabel("Dokładność")
        i = i +1;
        clear("ACC")
        saveas(gcf, "img/accuracy-per-class-" + tfun +"-"+ afun + ".png")
        close(gcf) 
    end
end

%%%%%%%%%%%%%%%%%%% Wykres wszystkie algorytmy - czułość %%%%%%%%%%%%%%%%%%%%%%%%%%%% 
training_functions = ["trainscg" "trainrp" "traincgb"];
activation_functions = ["tansig" "logsig"];

hold off
i = 1;
cont = {};
MM = zeros(2,1);
for tfun = training_functions
    for afun = activation_functions
        load(tfun +"_"+ afun);
        [M, I] = max(mean(RECALL, 2));
        if MM(1) < M
            MM = [M I];
        end

        plot(10:10:160,mean(RECALL, 2), 'LineWidth', 2);
        ylim([0.5 1]);
        cont{i} = tfun +"--" + afun;
        set(gca, 'Xtick', 10:10:160);
        xticklabels(labels)
        hold on
        i = i +1;
        clear("RECALL")
    end
end

plot(MM(2)*10, MM(1), '.', 'MarkerSize', 20)
cont{length(cont)+1}= "Punkt z największą czułością";
title(["Wykres czułości od ilości neuronów dla rożnych", ...
    "algorytmów uczących i funkcji aktywacji"])
xlabel("Ilość neuronów w warstwie ukrytej")
ylabel("Czułość")

legend(cont, 'Location', 'southwest')
saveas(gcf, 'img/recall-all.png')
close(gcf)

%%%%%%%%%%%%% Wykresy czułości per klasa dla algorytmów %%%%%%%%%%%%%%%%%%%
clear
clc
training_functions = ["trainscg" "trainrp" "traincgb"];
activation_functions = ["tansig" "logsig"];

i = 1;
cont = {};

for tfun = training_functions
    for afun = activation_functions
        load(tfun +"_"+ afun);
        plot(RECALL, 'LineWidth', 2);
        set(gca, 'Xtick', 1:1:16);
        xticklabels(labels)
        title(["Czułość dla każdej z klas dla kombinacji", ...
            (tfun +"--"+ afun)])
        ylim([0.5 1]);
        for j=[1:12]
            cont{j} = num2str(j);
        end
        
        leg = legend(cont,'Location', 'southeast');
        title(leg, "Klasy");
        leg.Title.Visible = 'on';
        xlabel("Ilość neuronów w warstwie ukrytej")
        ylabel("Czułość")
        i = i +1;
        clear("ACC")
        saveas(gcf, "img/recall-per-class-" + tfun +"-"+ afun + ".png")
        close(gcf) 
    end
end


