%% Start file 
spot_offset = T0; % Spot time in days
additional_days = 5 ; % Extra 5 days to avoid extrapolation
days_in_year = 365.25; % Average days in a year
T_values{1} = spot_offset/365;

%% Fix in dataset
for i = 2:length(priceAll) % Start from the second element to avoid out-of-bounds access
    % Check if the size of the double inside the cell is not 28x1
    if ~isequal(size(priceAll{i}), [28, 1])
        % Replace the current cell's content with the previous cell's content
        priceAll{i} = priceAll{i-1};
    end
    if ~isequal(size(T_valuesEnd{i}), [28, 1])
        T_valuesEnd{i} = T_valuesEnd{i-1};
    end
    if ~isequal(size(T_valuesAll{i}), [28, 1])
        T_valuesAll{i} = T_valuesAll{i-1};
    end
end
disp('All shorter matrices have been replaced with the previous matrix.');

delta_T = cell(size(T_values));
for index = 1:length(T_values)
    % Get the current cell's value
    T = T_values{index};
    if isempty(T)
       % Handle empty cells by assigning an empty array
       delta_T{index} = [];
    elseif length(T) == 1
        % If there is only one value, keep it unchanged
        delta_T{index} = T;
    else
        % Calculate differences for multiple values
        delta_T{index} = [T(1); diff(T)]; % Ensure vertical concatenation
    end
end
%% Single day plot
% day = 4826; % dagen vi vill plotta
% IntPrice = priceAll{day,1};
% T_values = T_valuesAll{day};
% last_entries = T_valuesEnd{day};
% T = last_entries;
% nodes = last_entries;
% yOIS_Actual = IntPrice(1:end); % test
% tol = 10^-10;
% maxIter = 10000;
% short_rate = yOIS_Actual(1);
% P0 = exp(-short_rate * (1 / 365));
% 
% for index = 1:length(T_values)
%     % Get the current cell's value
%     T = T_values{index};
%     if isempty(T)
%        % Handle empty cells by assigning an empty array
%        delta_T{index} = [];
%     elseif length(T) == 1
%         % If there is only one value, keep it unchanged
%         delta_T{index} = T;
%     else
%         % Calculate differences for multiple values
%         delta_T{index} = [T(1); diff(T)]; % Ensure vertical concatenation
%     end
% end
% 
% methods = ["linear_discount", "linear_spot", "raw", "log_rates"];
% 
% for method = (methods)
%     [discountFactors, iterations, F, f_f, y_error] = NewtonRaphson(yOIS_Actual, delta_T, T_values, tol,maxIter, nodes, P0, last_entries, method);
%     for i = 1:5 
%         f_f(:,i) = f_f(:,6);
%     end
%     plot_f_f(f_f(1:3650), method);
% end
% test = exp(-discountFactors);
%% 3D plots
method = "linear_discount";
f_toti = zeros(4826, 3650);
y_error1 = zeros(4826, 27);
for i = 1:4826
    disp(i);
    IntPrice = priceAll{i,1};
    % T = last_entries;
    T = T_valuesEnd{i};
    yOIS_Actual = IntPrice(1:end);
    tol = 10^-10;
    maxIter = 10000;
    short_rate = IntPrice(1);
    P0 = exp(- short_rate * (1 / 365));
    nodes = T_valuesEnd{i};
    T_values = T_valuesAll{i};
    % delta_T = T_values3{i};
    %deltaT replacement
    delta_T = cell(size(T_values));
    for index = 1:length(T_values)
        % Get the current cell's value
        T = T_values{index};
        if isempty(T)
           % Handle empty cells by assigning an empty array
           delta_T{index} = [];
        elseif length(T) == 1
            % If there is only one value, keep it unchanged
            delta_T{index} = T;
        else
            % Calculate differences for multiple values
            delta_T{index} = [T(1); diff(T)]; % Ensure vertical concatenation
        end
    end
    last_entries = T_valuesEnd{i};
    [discountFactors, iterations, F, f_f, y_error] = NewtonRaphson(yOIS_Actual, delta_T, T_values, tol, maxIter, nodes, P0, last_entries, method);
    f_toti(i, :) = f_f(1:3650);
    y_error1(i,:) = y_error;
end

for i = 1:5 
    f_toti(:,i) = f_toti(:,6);
end
f_toti(1374,:) = f_toti(1375,:); 
plot_3D_rates_corrected(f_toti(1:4826,:), times(1:4826), method);
error = mean(y_error1(:,3));

%% 3D plot with smoothing
% sigma = 10; % smoothparameter
% f_toti_smoothed = imgaussfilt(f_toti(1:4826,:), sigma);
% plot_3D_rates_corrected(f_toti_smoothed, times(1:4826), method);

%% Main function
function [discountFactors, iterations, F, f_f, y_error] = NewtonRaphson(yOIS_Actual, delta_T, T_values, tol, maxIter, nodes, P0, lastentries, method)

    % Inputs:
    % yOIS_Actual - Vector of actual market yOIS values (28x1)
    % delta_T - Cell array of cash flow times (1x28)
    % T_values - Cell array of tenors corresponding to each delta_T (1x28)
    % tol - Convergence tolerance for the residual norm
    % maxIter - Maximum number of iterations
    
    % Define the time-to-maturity for the discount factors
    T = lastentries;
    
    % Choose a decay rate alpha
    alpha = 0.01; % godtycklig
    
    % Calculate the discount factors using exponential decay
    discountFactors = exp(-alpha * T);
    discountFactors = discountFactors';

    % Initialize P with the same structure as delta_T
    P = delta_T;
    P{1} = P0;
    if method == "log_rates"
        disp("nu")
    end

    % Loop over each OIS contract
    for i = 2:length(P)    
        globalTenors = nodes;
            for j = 1:length(P{i})
                tau = T_values{i}(j);
                tau_i_index = find(globalTenors <= tau, 1, 'last'); % Largest tau_i <= tau
                tau_i1_index = find(globalTenors >= tau, 1, 'first'); % Smallest tau_i1 >= tau

                % Get corresponding tau_i, tau_i1, d_i, and d_i1
                tau_i = globalTenors(tau_i_index);
                tau_i1 = globalTenors(tau_i1_index);

                if tau == tau_i 
                    P{i}(j) = discountFactors(tau_i_index);
                else
                d_i = discountFactors(tau_i_index);
                d_i1 = discountFactors(tau_i1_index);
    
                % Calculate intermediate discount factor using linear interpolation
                [P{i}(j),~] = interpolateDiscountFactors(tau, tau_i, tau_i1, d_i, d_i1, method);
                end
            end
    end

    % Iterate until convergence or max iterations reached
    for iter = 1:maxIter
        % Step 1: Compute yOIS using current discount factors
        yOIS = Compute_yOIS(P, delta_T, P0);

        % Step 2: Compute residuals (F)
        F = Get_F(yOIS, yOIS_Actual, iter);
        
        % Step 3: Compute the Jacobian matrix (partial derivatives)
        J = Compute_Jacobian(P, delta_T, discountFactors, T_values, nodes, length(discountFactors), P0, method);
        
        % Step 4: Solve for update step (Delta x)
        Delta_x = J \ F;
        
        % Step 5: Initialize gamma and perform backtracking
        gamma = 1;
        while true
            % Compute tentative updated discount factors 
            discountFactors_new = discountFactors + gamma * Delta_x;
            
            for i = 2:length(P)
                globalTenors = nodes; % Extract global tenor values
                
                % If there are intermediate cash flows, calculate their discount factors
                for j = 1:length(P{i})
       
                    tau = T_values{i}(j);
                    tau_i_index = find(globalTenors <= tau, 1, 'last'); % Largest tau_i <= tau
                    tau_i1_index = find(globalTenors >= tau, 1, 'first'); % Smallest tau_i1 >= tau

                    % Get corresponding tau_i, tau_i1, d_i, and d_i1
                    tau_i = globalTenors(tau_i_index);
                    tau_i1 = globalTenors(tau_i1_index);

                    if tau == tau_i 
                        P{i}(j) = discountFactors_new(tau_i_index);
                    else
                    
                    d_i = discountFactors_new(tau_i_index);
                    d_i1 = discountFactors_new(tau_i1_index);
        
                    % Calculate intermediate discount factor using linear interpolation
                    [P{i}(j),~] = interpolateDiscountFactors(tau, tau_i, tau_i1, d_i, d_i1, method);
                    end
                end
            end
            
            % Compute new residuals
            F_new = Get_F(Compute_yOIS(P, delta_T, P0), yOIS_Actual, iter);
            disp(norm(F,2));
            disp(norm(F_new,2));
           
             if norm(F_new,2) <= norm(F,2)
                % Sufficient decrease, accept the step
                discountFactors = discountFactors_new;
                break;
            else
                % Reduce gamma
                gamma = gamma / 2;
             end
        end
        
        % Step 6: Check convergence based on the norm of Delta_x
        if norm(Delta_x,2) < tol
            fprintf('Converged after %d iterations. Step norm: %e\n', iter, norm(Delta_x,2));
            iterations = iter; % Output the number of iterations
            if method == "linear_discount"
                f_f = Ffunc_LinearOnDis(lastentries, discountFactors);
                y_error = yOIS_Actual(2:end) - yOIS;
            elseif method == "raw"
                r_i = - log(discountFactors);
                r_i = r_i ./ lastentries';
                f_f = Ffunc_Raw(lastentries, r_i);
                y_error = yOIS_Actual(2:end) - yOIS;
            elseif method == "linear_spot"
                r_i = - log(discountFactors);
                r_i = r_i ./ lastentries';
                f_f = Ffunc_LinearOnSpot(lastentries, r_i);
                y_error = yOIS_Actual(2:end) - yOIS;
            elseif method == "log_rates"
                r_i = - log(discountFactors);
                r_i = r_i ./ lastentries';
                f_f = Ffunc_LinearOnLog(lastentries, r_i);
                y_error = yOIS_Actual(2:end) - yOIS;
            end
            return;
        end
    end

% If max iterations are reached without convergence
fprintf('Did not converge within %d iterations. Final step norm: %e\n', maxIter, norm(Delta_x,2));
iterations = maxIter; % Output the number of iterations
 
end
%% Calculating Loocv
% ta bort 3an 
% function [f_k] = Loocv(tau_i, d_i)
% 
% end
%% fk calculations final
function [f_k] = Ffunc_LinearOnDis(tau_i, d_i)
    r_k = r_function_dis(tau_i, d_i);
    f_k = f_function_dis(tau_i, r_k);
end

function [r_k] = r_function_dis(tau_i, d_i)
    count = 2;
    total_days = tau_i(end) - tau_i(1) + 1 / 365;
	tau_k = linspace(tau_i(1), tau_i(end), total_days * 365 + 1);
    r_k = zeros(size(tau_k));
    
    for k = 1:(total_days * 365 + 1)
        r_k(k) =  (-1 / tau_k(k)) * log( ((tau_k(k) - tau_i(count-1)) / (tau_i(count) - tau_i(count-1)) * d_i(count)) + ...
                                        (tau_i(count) - tau_k(k)) / (tau_i(count) - tau_i(count-1)) * d_i(count-1) );
        if (abs(tau_k(k) - tau_i(count)) < 1/366)
            count = count + 1;
        end    
    end
    
end 

function [f_k] = f_function_dis(tau_i, r_k)

    total_days = tau_i(end) - tau_i(1) + 1/365;
    tau_k = linspace(tau_i(1), tau_i(end), total_days * 365 + 1);
    f_k = zeros(size(r_k));
    
    for k = 1:(total_days * 365)
        f_k(k) = (r_k(k+1) * tau_k(k+1) - r_k(k) * tau_k(k)) * 365;
    end
    
    if tau_i(1) == 0
       f_k(1) = f_k(2); 
    end

end 

function[f_k] = Ffunc_Raw(tau_i, r_i)
    
    total_days = tau_i(end) - tau_i(1) + 1/365;
    tau_k = linspace(tau_i(1), tau_i(end), total_days * 365 + 1);
    f_k = zeros(size(tau_k));
    count = 2;
    
    for k = 1:(total_days * 365)
        f_k(k) = (tau_i(count) / (tau_i(count) - tau_i(count - 1))) * r_i(count) - ...
                (tau_i(count - 1) / (tau_i(count) - tau_i(count - 1))) * r_i(count - 1);
        if (abs(tau_k(k) - tau_i(count)) < 1/366)
            count = count + 1;
        end
    end
end

% Linear spot
function [f_k] = Ffunc_LinearOnSpot(tau_i, r_i)
    r_k = r_function_spot(tau_i, r_i);
    f_k = f_function_spot(tau_i, r_k);
end

function [r_k] = r_function_spot(tau_i, r_i)
    count = 2;
    total_days = tau_i(end) - tau_i(1) + 1 / 365;
    tau_k = linspace(tau_i(1), tau_i(end), total_days * 365 + 1);
    r_k = zeros(size(tau_k));
    for k = 1:(total_days * 365 + 1)
        r_k(k) = (tau_k(k) - tau_i(count-1)) / (tau_i(count) - tau_i(count-1)) * r_i(count) + ...
        (tau_i(count) - tau_k(k)) / (tau_i(count) - tau_i(count-1)) * r_i(count-1);
        if (abs(tau_k(k) - tau_i(count)) < 1/366)
            count = count + 1;
        end
    end
end

function [f_k] = f_function_spot(tau_i, r_k)
    total_days = tau_i(end) - tau_i(1) + 1/365;
    tau_k = linspace(tau_i(1), tau_i(end), total_days * 365 + 1);
    f_k = zeros(size(r_k));
    for k = 1:(total_days * 365)
        f_k(k) = (r_k(k+1) * tau_k(k+1) - r_k(k) * tau_k(k)) * 365;
    end
end

% Linear log
function [f_k] = Ffunc_LinearOnLog(tau_i, r_i)
    r_k = r_function_log(tau_i, r_i);
    f_k = f_function_log(tau_i, r_k);
end
function [r_k] = r_function_log(tau_i, r_i)
    count = 2;
    total_days = tau_i(end) - tau_i(1) + 1 / 365;
    tau_k = linspace(tau_i(1), tau_i(end), total_days * 365 + 1);
    r_k = zeros(size(tau_k));
    for k = 1:1:(total_days * 365 + 1)
        r_k(k) = r_i(count)^( (tau_k(k)-tau_i(count-1)) / (tau_i(count)-tau_i(count-1)) ) * ...
        r_i(count-1)^( (tau_i(count)-tau_k(k)) / (tau_i(count)-tau_i(count-1)) );
        if (abs(tau_k(k) - tau_i(count)) < 1/366)
            count = count + 1;
        end
    end
end

function [f_k] = f_function_log(tau_i, r_k)
    total_days = tau_i(end) - tau_i(1) + 1/365;
    tau_k = linspace(tau_i(1), tau_i(end), total_days * 365 + 1);
    f_k = zeros(size(r_k));
    for k = 1:(total_days * 365)
        f_k(k) = (r_k(k+1) * tau_k(k+1) - r_k(k) * tau_k(k)) * 365;
    end
end

function plot_f_f(f_f, method)

    % Check if the input is valid
    if nargin < 2
        error('Both the vector f_f and the method name must be provided as input.');
    end
    if ~isvector(f_f)
        error('Input f_f must be a vector.');
    end
    
    % Create the x-axis values as "Maturity (years)"
    maturity_years = (1:length(f_f)) / 365;

    % Plot the vector f_f
    figure; % Open a new figure
    plot(maturity_years, f_f, 'LineWidth', 2); % Plot with maturity in years
    if method == "raw"
        title('Plot of forward rates with method: Raw interpolation');
    elseif method == "linear_discount"
        title('Plot of forward rates with method: Linear on Discount Factors');
    elseif method == "linear_spot"
        title('Plot of forward rates with method: Linear on Spot Rates');
    elseif method == "log_rates"
        title('Plot of forward rates with method: Linear on Logarithm of Rates');
    end
    xlabel('Maturity (years)');
    ylabel('Forward rate (%)');
    y_limits = ylim; % Get current y-axis limits
    ylim([min(0, y_limits(1)), y_limits(2)]);
end

function plot3DCurve(times, maturities, fAll)
% --- Construct a matrix from the cell array ---
% Rows = different dates
% Columns = different maturities
fMatrix = fAll;

% --- Create grids for plotting ---
% Let X = maturities, Y = times (or vice versa, your choice)
[X, Y] = meshgrid(maturities, times);

% --- 3D surface plot ---
figure;
surf(X, Y, fMatrix');   % If dimension mismatch, use surf(X, Y, fMatrix') or transpose X/Y/fMatrix
shading interp;        % smooth shading
colormap(jet);
colorbar;
xlabel('Maturity (Years)');
ylabel('Date (Serial Days)');
zlabel('Forward Rate (%)');
title('Synthetic Forward Rate Surface');

% If your "times" are serial date numbers, format them nicely on the Y-axis
datetick('y','yyyy');  % convert Y tick marks to readable years

axis tight;
view([-45 30]);        % adjust 3D view angle
end

function plot_3D_rates2(f_toti, times)
    % plot_3D_rates - Create a 3D plot for the rate matrix
    %
    % Syntax:
    %   plot_3D_rates(f_toti, times)
    %
    % Inputs:
    %   f_toti - A 3000x3650 matrix where each row represents rates for a specific day.
    %   times - A vector representing the dates (in serial date number format) for the rows of f_toti.
    %
    % Example:
    %   plot_3D_rates(f_toti, times);

    % Validate inputs
    if nargin < 2
        error('Both f_toti and times must be provided.');
    end
    if size(f_toti, 2) ~= 3650
        error('f_toti must have 3650 columns representing maturities up to 10 years.');
    end
    if size(f_toti, 1) ~= length(times)
        error('The number of rows in f_toti must match the length of times.');
    end

    % Convert times to datetime format
    times_datetime = datetime(times, 'ConvertFrom', 'datenum'); % Convert to datetime

    % Define maturities in years
    maturities = (1:3650) / 365; % Convert maturities to years
    [Maturities, Times] = meshgrid(maturities, times_datetime); % Create a grid

    % Plot the surface
    figure;
    surf(Maturities, Times, f_toti, 'EdgeColor', 'none'); % Surface plot
    colorbar; % Add a colorbar to indicate rates
    title('3D Plot of Forward Rates');
    xlabel('Maturity (years)');
    ylabel('Time (date)');
    zlabel('Rate');
    datetick('y', 'yyyy-mm-dd', 'keeplimits'); % Format y-axis to show dates
    view(45, 30); % Adjust the view angle for better visibility
    grid on;
end
function plot_3D_rates_corrected(f_toti, times, method)
    % plot_3D_rates_corrected - Create a 3D plot for the rate matrix with
    % maturity on the x-axis, date on the y-axis, and rate on the z-axis.
    %
    % Syntax:
    %   plot_3D_rates_corrected(f_toti, times)
    %
    % Inputs:
    %   f_toti - A 3000x3650 matrix where each row represents rates for a specific day.
    %   times - A vector representing the dates (in serial date number format) for the rows of f_toti.
    %
    % Example:
    %   plot_3D_rates_corrected(f_toti, times);

    % Validate inputs
    if nargin < 2
        error('Both f_toti and times must be provided.');
    end
    if size(f_toti, 2) ~= 3650
        error('f_toti must have 3650 columns representing maturities up to 10 years.');
    end
    if size(f_toti, 1) ~= length(times)
        error('The number of rows in f_toti must match the length of times.');
    end

    % Convert times to datetime format
    times_datetime = datetime(times, 'ConvertFrom', 'datenum'); % Convert to datetime

    % Define maturities in years
    maturities = (1:3650) / 365; % Convert maturities to years
    [Maturities, Dates] = meshgrid(maturities, times_datetime); % Create a grid

    % Plot the surface
    figure;
    shading interp;
    colormap(jet);
    surf(Maturities, Dates, f_toti, 'EdgeColor', 'none'); % Surface plot
    colorbar; % Add a colorbar to indicate rates
    if method == "raw"
        title('3D Plot of Forward Rates Raw Interpolation');
    elseif method == "linear_discount"
        title('3D Plot of Forward Rates Linear on Discount Factors');
    elseif method == "linear_spot"
        title('3D Plot of Forward Rates using Linear on Spot Rates');
    end
    xlabel('Maturity (years)');
    ylabel('Date');
    zlabel('Rate');
    datetick('y', 'yyyy-mm-dd', 'keeplimits'); % Format y-axis to show dates
    view(-45, 30); % Adjust the view for the desired orientation
    grid on;
end


%% Function to calculate the yOIS
function yOIS = Compute_yOIS(P, delta_T, P0)
    % Inputs:
    % P - Cell array of discount factors, matching the format of delta_T (1x28)
    % delta_T - Cell array of cash flow times (1x28)
    
    % Number of instruments
    numInstruments = length(P)-1;
    
    % Initialize yOIS vector
    yOIS = zeros(numInstruments, 1);
    
    % Compute yOIS for each instrument
    for i = 1:numInstruments
        % Get discount factors and delta_T for current instrument
        P_i = P{i+1}; % Discount factors for instrument i
        delta_T_i = delta_T{i+1}; % Cash flow times for instrument i
        
        % Determine K (number of cash flows)
        K = length(P_i);
        
        % Compute yOIS for instrument i
        yOIS(i) = (P0 - P_i(K)) / sum(delta_T_i .* P_i);
    end
end

%% Function to calculate F
function F = Get_F(yOIS, yOIS_Actual, iter)
    F = zeros(28,1);
    % test = zeros(28,1);
    for i = 1:length(yOIS)
        % if length(yOIS_Actual) <= 26 
        %     if length(yOIS_Actual) == 25
        %         if i == 25
        %             F(i+1) = F(i);
        %             F(i+2) = F(i+1);
        %             F(i+3) = F(i+1);
        %             break;
        %         else
        %             F(i+1) = yOIS_Actual(i) - yOIS(i);
        %         end
        %     end
        %     % F(i+1) = yOIS_Actual(i) - yOIS(i);
        %     if i == 26
        %         F(i+1) = F(i);
        %         F(i+2) = F(i+1);
        %         break;
        %     else
        %          F(i+1) = yOIS_Actual(i) - yOIS(i);
        %     end
        % else
            % F(i+1) = yOIS_Actual(i) - yOIS(i); %old
            F(i+1) = yOIS_Actual(i) - yOIS(i);
        % end      
    end
end


%% Functions calulating tau (funkar, använd denna)
% function tau = get_tau(T_values)
%     tau = cellfun(@(x) x(end), T_values, 'UniformOutput', true);
% end
%% Functions Calculating the discretized rates
function r_k = r_k_linear_on_disc(tau_k, tau_i, tau_i1, d_i, d_i1)
    r_k = - (1/tau_k) * log(((tau_k - tau_i) / (tau_i1 - tau_i)) * d_i1 + ((tau_i1 - tau_k) / (tau_i1 - tau_i)) * d_i);
end

% Linear on Spot Rates
function r_k = r_k_linear_spot(tau_k, tau_i, tau_i1, r_i, r_i1)
    r_k = ((tau_k - tau_i) / (tau_i1 - tau_i)) * r_i1 + ((tau_i1 - tau_k) / (tau_i1 - tau_i)) * r_i;
end

% Raw Interpolation
function r_k = r_k_raw(tau_k, tau_i, tau_i1, r_i, r_i1)
    term1 = ((tau_k - tau_i) / (tau_i1 - tau_i)) * ((tau_i1 - tau_k) / tau_k) * r_i1;
    term2 = ((tau_i1 - tau_k) / (tau_i1 - tau_i)) * (tau_k / tau_i) * r_i;
    r_k = term1 + term2;
end

% Linear on the Logarithm of Rates
function r_k = r_k_linear_log(tau_k, tau_i, tau_i1, r_i, r_i1)
    term1 = r_i1 ^ ((tau_k - tau_i) / (tau_i1 - tau_i));
    term2 = r_i ^ ((tau_i1 - tau_k) / (tau_i1 - tau_i));
    r_k = term1 * term2;
end


  


%% Function calculating Jacobian

function J = Compute_Jacobian(P, delta_T, discountFactors, T_values, nodes, num_variables, P0, method)
       % Inputs:
    %   - P: Cell array of discount factor sets for OIS contracts.
    %   - delta_T: Cell array of time intervals for OIS contracts.
    %   - discountFactors: Global discount factors for all tenors.
    %   - T_values: Cell array of cash flow times for OIS contracts.
    %   - nodes: Global tenor points.
    %
    % Output:
    %   - Yoisder: Cell array of Jacobian contributions for each OIS contract.

    % Initialize Jacobian component
    num_OIS = length(P)-1;
    %Yoisder = cell(1, num_OIS);
    %globalTenors = nodes;
    J = zeros(28, 28);  % Preallocate Jacobian matrix
    J(1,1) = 1; %T0 derivative
    % --- Case 1: Contribution to the first two elements ---

   
    
    % Compute case 1 contribution for all 28 contracts
    case1_contributions = cell(1, num_OIS); % Preallocate
    for n = 1:num_OIS
        P_n = P{n+1};
        delta_T_n = delta_T{n+1};
        case1_contributions{n} = zeros(2, 1);
        sum_delta_P = sum(delta_T_n .* P_n);
        % case1_contributions{n} =  [(1 / sum_delta_P) * d_inner(1); ...
        %      (1 / sum_delta_P) * d_inner(2)];
        case1_contributions{n} = (1 / sum_delta_P);
    end

     %case 1 
    for n = 1:num_OIS
        J(n+1,1) = J(n+1,1) + case1_contributions{n};
    end

    % --- Case 3: Contribution to the last two elements ---
    %case3_contributions = cell(1, num_OIS); % Preallocate
    for n = 2:num_OIS+1
        P_n = P{n};
        delta_T_n = delta_T{n};
        tau = T_values{n}(end);
        tau_i_index = find(nodes <= tau, 1, 'last');
        tau_i1_index = find(nodes >= tau, 1, 'first');
        tau_i = nodes(tau_i_index);
        tau_i1 = nodes(tau_i1_index);
        sum_delta_P = sum(delta_T_n .* P_n);
        numerator = P0 - P_n(end);
        denominator = sum_delta_P^2;
        d_i = discountFactors(tau_i_index);
        d_i1 = discountFactors(tau_i1_index);
        
        if tau == tau_i
            J(n, tau_i_index) = J(n, tau_i_index) + (-((1 / sum_delta_P) + delta_T_n(end) * numerator / denominator));
        else
            [~,d_inner] = interpolateDiscountFactors(tau, tau_i, tau_i1, d_i, d_i1, method);
            %d_inner = Compute_Inner_Derivatives(tau, tau_i, tau_i1);
            J(n, tau_i_index) = J(n, tau_i_index) + (-((1 / sum_delta_P) + delta_T_n(end) * numerator / denominator) * d_inner(1));
            J(n, tau_i1_index) = J(n, tau_i1_index) + (-((1 / sum_delta_P) + delta_T_n(end) * numerator / denominator) * d_inner(2));
        end

    end

   
   for n = 2:num_OIS+1
        P_n = P{n};
        delta_T_n = delta_T{n};
        K = length(P_n);
        numerator = P0 - P_n(end);
        denominator = sum(delta_T_n .* P_n)^2;

        % Case 2: Original logic
        for j = 1:K-1
            tau = T_values{n}(j);
            tau_i_index = find(nodes <= tau, 1, 'last');
            tau_i1_index = find(nodes >= tau, 1, 'first');
            tau_i = nodes(tau_i_index);
            tau_i1 = nodes(tau_i1_index);
            d_i = discountFactors(tau_i_index);
            d_i1 = discountFactors(tau_i1_index);
            if tau == tau_i
                dF_dx = -delta_T_n(j) * numerator / denominator;
                J(n,tau_i_index) = J(n,tau_i_index) + dF_dx;
            else
                [~,d_inner] = interpolateDiscountFactors(tau, tau_i, tau_i1, d_i, d_i1, method);
                %d_inner = Compute_Inner_Derivatives(tau, tau_i, tau_i1);
                dF_dx_i = -delta_T_n(j) * numerator / denominator * d_inner(1);
                dF_dx_i1 = -delta_T_n(j) * numerator / denominator * d_inner(2);
                J(n,tau_i_index) = J(n,tau_i_index) + dF_dx_i;
                J(n,tau_i1_index) = J(n,tau_i1_index) + dF_dx_i1;  
            end

          
        end

   
    end


  
   
end







%% Functions calculating the forward rate
% Linear on Discount Factors
function fk = f_k_linear_dis(tau_k, tau_k1, tau_i, tau_i1, d_i, d_i1)
    Top = log((tau_k - tau_i) / (tau_i1 - tau_i) * d_i1 + (tau_i1 - tau_k) / (tau_i1 - tau_i) * d_i) - log((tau_k1 - tau_i) / (tau_i1 - tau_i) * d_i1 + (tau_i1 - tau_k1) / (tau_i1 - tau_i) * d_i);
    Bottom = tau_k1 - tau_k;
    fk = Top / Bottom;
end

% Linear on Spot Rates
function fk = f_k_linear_spot(tau_k, tau_k1, tau_i, tau_i1, r_i, r_i1)
    term1 = tau_k1 * (((tau_k1 - tau_i) / (tau_i1 - tau_i)) * r_i1 + ((tau_i1 - tau_k1) / (tau_i1 - tau_i)) * r_i);
    term2 = tau_k * (((tau_k - tau_i) / (tau_i1 - tau_i)) * r_i1 + ((tau_i1 - tau_k) / (tau_i1 - tau_i)) * r_i) * ((tau_k1 - tau_k) / (r));
    fk = (term1 - term2) / (tau_k1 - tau_k);
end

% Raw Interpolation
function fk = f_k_raw(tau_k, tau_k1, tau_i, tau_i1, r_i, r_i1)
    term1 = (((tau_k1 - tau_i) / (tau_i1 - tau_i)) * (tau_i1 / tau_k1) * r_i1 + ((tau_i1 - tau_k1) / (tau_i1 - tau_i)) * (tau_i / tau_k1) * r_i) * tau_k1;
    term2 = (((tau_k - tau_i) / (tau_i1 - tau_i)) * (tau_i1 / tau_k) * r_i1 + ((tau_i1 - tau_k) / (tau_i1 - tau_i)) * (tau_i / tau_k) * r_i) * tau_k;
    fk = (term1 - term2) / (tau_k1 - tau_k);
end

% Linear on the Logarithm of Rates
function fk = f_k_linear_log(tau_k, tau_k1, tau_i, tau_i1, r_i, r_i1)
    term1 = (((tau_k1 - tau_i) / (tau_i1 - tau_i)) * (tau_i1 / tau_k1) * r_i1 + ((tau_i1 - tau_k1) / (tau_i1 - tau_i)) * (tau_i / tau_k1) * r_i) * tau_k1;
    term2 = (((tau_k - tau_i) / (tau_i1 - tau_i)) * (tau_i1 / tau_k) * r_i1 + ((tau_i1 - tau_k) / (tau_i1 - tau_i)) * (tau_i / tau_k) * r_i) * tau_k;
    fk = (term1 - term2) / (tau_k1 - tau_k);
end

%% Functions calculating discountfactors for intermediate cash flows

function d_tau = Compute_Linear_Discount(tau, tau_i, tau_i1, d_i, d_ip1)
    % Compute the discount factor using the formula
    d_tau = ((tau - tau_i) / (tau_i1 - tau_i)) * d_ip1 + ((tau_i1 - tau) / (tau_i1 - tau_i)) * d_i;
end

%% Functions calculating the inner derivatives
function inner_der = Compute_Inner_Derivatives(tau, tau_i, tau_i1)
    % Calculate the inner derivatives with respect to d_i and d_i1
    d_tau_di = (tau_i1 - tau) / (tau_i1 - tau_i); % Derivative w.r.t. d_i
    d_tau_di1 = (tau - tau_i) / (tau_i1 - tau_i);  % Derivative w.r.t. d_i1
    inner_der = [d_tau_di; d_tau_di1];
end

%%Interpolation Method
function [discountFactor, innerDerivatives] = interpolateDiscountFactors(tau, tau_i, tau_i1, d_i, d_i1, method)
% Function to calculate interpolated discount factor and inner derivatives
% for different interpolation methods.

% Inputs:
% tau      - Target time to maturity
% tau_i    - Maturity time of node i
% tau_ip1  - Maturity time of node i+1
% di       - Discount factor at node i
% dip1     - Discount factor at node i+1
% method   - Interpolation method: 'linear_spot', 'raw', 'log_rates', 'linear_discount'

    switch method
        case 'linear_discount'
            % Linear on discount factors
            weight_i = (tau_i1 - tau) / (tau_i1 - tau_i);
            weight_ip1 = (tau - tau_i) / (tau_i1 - tau_i);
            discountFactor = d_i * weight_i + d_i1 * weight_ip1;
            innerDerivatives = [weight_i, weight_ip1];
    
        case 'linear_spot'
            % Linear on spot rates
            denom = tau_i1-tau_i;
            discountFactor = exp(-tau * ((tau-tau_i)/denom) * (-log(d_i1) / tau_i1) -tau * ((tau_i1- tau)/denom) * (-log(d_i) / tau_i));
            innerDerivatives = [tau * discountFactor * (tau_i1-tau)/(denom * tau_i * d_i), tau * discountFactor * (tau-tau_i)/(denom * tau_i1 * d_i1) ];
    
        case 'raw'
            % Raw interpolation
            ln_di = log(d_i);
            ln_dip1 = log(d_i1);
            ln_d_tau = (tau_i1 - tau) / (tau_i1 - tau_i) * ln_di + (tau - tau_i) / (tau_i1 - tau_i) * ln_dip1;
            discountFactor = exp(ln_d_tau);
            partial_di = discountFactor * (tau_i1 - tau) / (tau_i1 - tau_i) / d_i;
            partial_dip1 = discountFactor * (tau - tau_i) / (tau_i1 - tau_i) / d_i1;
            innerDerivatives = [partial_di, partial_dip1];
    
        case 'log_rates'
            Linear on logarithm of rates
            denom = tau_i1-tau_i;
            r_tau = (-log(d_i1)/tau_i1)^((tau-tau_i)/denom) * (-log(d_i)/tau_i)^((tau_i1-tau)/denom);
            discountFactor = exp(-tau * r_tau);
            inner1 = tau * discountFactor * ((tau_i1-tau)/denom) * (1/(d_i * tau_i)) * (-log(d_i)/tau_i)^(((tau_i1-tau)/denom)-1);
            inner2 = tau * discountFactor * ((tau-tau_i)/denom) * (1/(d_i1 * tau_i1)) * (-log(d_i1)/tau_i1)^(((tau-tau_i)/denom)-1);
            innerDerivatives = [inner1,inner2];
            
        otherwise
            error('Invalid interpolation method specified.');
    end
end
