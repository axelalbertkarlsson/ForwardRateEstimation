clc;
data = load('matrix_to_carl.mat');

errors_matrix = data.errors_matrix;
labels_for_models = data.labels_of_models;

% True means comparing every 3W method with eachother and onwards. 
% False means comparing Kalman 3W with the other methods for 3W respecvtively and onwards.
choice_of_comparison = true;

if(choice_of_comparison)
    num_models_per_group = 4;
    num_groups = 3;
    
    alpha_matrix = zeros(num_models_per_group, num_models_per_group, num_groups);
    boolean_matrix = zeros(num_models_per_group, num_models_per_group, num_groups);
    
    group_names = {'3W', '1Y', '5Y'};
    
    for group = 1:num_groups
        if group == 1
            group_indices = [1, 4, 5, 6]; 
        elseif group == 2
            group_indices = [2, 7, 8, 9]; 
        elseif group == 3
            group_indices = [3, 10, 11, 12]; 
        end
    
        group_labels = labels_for_models(group_indices);
        group_errors = errors_matrix(:, group_indices); 
    
        for i = 1:num_models_per_group
            for j = 1:num_models_per_group
                if i ~= j 
                    error_model1 = group_errors(:, i);
                    error_model2 = group_errors(:, j);
                    label_error_model1 = group_labels{i};
                    label_error_model2 = group_labels{j};
    
                    [alpha, statistically_significant] = stattesting(error_model1, error_model2, label_error_model1, label_error_model2);
    
                    alpha_matrix(i, j, group) = alpha;
                    boolean_matrix(i, j, group) = statistically_significant;
                end
            end
        end
    
        figure(group);
        clf; 
        imagesc(alpha_matrix(:, :, group)); 
        colormap('Winter');
        colorbar;

        formatted_labels = strrep(group_labels, '_', ' ');
        xticks(1:num_models_per_group);
        yticks(1:num_models_per_group);
        xticklabels(group_labels);
        yticklabels(group_labels);
        ax = gca;
        ax.TickLabelInterpreter = 'none';
        xlabel('Models');
        ylabel('Models');
        title(sprintf('Probability of Outperformance for %s (Significance Level = 0.05)', group_names{group}));
    
        for i = 1:num_models_per_group
            for j = 1:num_models_per_group
                text(j, i, sprintf('%.f%%', 100*alpha_matrix(i, j, group)), ...
                    'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'middle', ...
                    'Color', 'white', 'FontSize', 10);
                if boolean_matrix(i, j, group) == 0
                    rectangle('Position', [j-0.5, i-0.5, 1, 1], ...
                              'EdgeColor', 'red', ...
                              'LineWidth', 2);
                end
            end
        end
    end

else
    for i = 1:3
        index = i*3;
        for j = index+1:index+3
            error_model1 = errors_matrix(:, i);
            error_model2 = errors_matrix(:, j);
            label_error_model1 = labels_for_models(i);
            label_error_model2 = labels_for_models(j);
            stattesting(error_model1, error_model2, label_error_model1, label_error_model2);
        end
    end
end

function [alpha, statistically_significant] = stattesting(error_model1, error_model2, label_error_model1, label_error_model2)
    T_1 = size(error_model1,1);
    T_2 = size(error_model2,1);
    if(T_1 == T_2)
        T = T_1;
    else
        error("Error: Time periods do not match for the models");
    end
    d = error_model1 - error_model2; %% Given error is MSE between realized and derived prices beforehand
    d_bar = mean(d); %%% MSE
    s= sqrt((1/(T-1))*sum((d-d_bar).^2));
    %s=sqrt(var(d));
    z = (d_bar*sqrt(T))/(s);
    right_side = (1-normcdf(abs(z))); %%% two-sided p_value
    significant_level = 0.01;
    left_side = significant_level;
    alpha = normcdf(-z);

    fprintf('Number of observations (T): %d\n', T);
    fprintf('Mean Squared Error (d_bar): %.8f\n', d_bar);
    fprintf('Sample volatility (s): %.8f\n', s);
    fprintf('Test statistic (z): %.4f\n', z);
    fprintf('Alpha: %.f%%\n', alpha*100);
    fprintf('Right side (p-value): %.4f\n', right_side);
    fprintf('Left side (significant_level): %.2f\n\n', left_side);
    
    fprintf('The probability that %s is better than %s: %.f%%\n',label_error_model1, label_error_model2, alpha*100);

    if right_side < left_side
        fprintf('Result: The comparison is statistically significant (p-value < significant_level)\n\n');
        statistically_significant = 1;
    else
        fprintf('Result: The comparison is not statistically significant (p-value >= significant_level)\n\n');
        statistically_significant = 0;
    end
end