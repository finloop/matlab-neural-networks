load("RES.mat")

classes = unique(table2array(RESULTS(:, 2)));

hold on
for class = classes'
    t = RESULTS(table2array(RESULTS(:, 2)) == class, :);
    plot(table2array(t(:, 1)), table2array(t(:, 3)), 'LineWidth', 1)
end

title("Wykres dokładności w zalezności od maksymalnej ilości rozgałęzień")
ylabel("Dokładność")
xlabel("Maksymalna ilośc rozgałęzień (MaxSplits)")
legend(classes, 'Location', 'southeast')
ylim([0 1])
hold off
saveas(gcf, 'img/acc.png')

RESULTS.Properties.VariableNames = ["MaxNumSplits" "Tree" "Acuracy"];
writetable(sortrows(RESULTS, [3], 'descend'), "res.csv")