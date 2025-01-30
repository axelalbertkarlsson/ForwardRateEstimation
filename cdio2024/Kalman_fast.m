load Copy_of_OISdata.mat

close all
TT= length(times);
K = 5;

D = 0.99*D;


% K är antalet iterationer
xAll = cell(length(times), 1);
wAll = cell(length(times), 1);
vAll = cell(length(times), 1);
A_overlineAll = cell(length(times), 1);
P_overlineAll = cell(length(times), 1);
a_overlineAll = cell(length(times), 1);
delta_wAll = cell(length(times), 1);
delta_vAll = cell(length(times), 1);
HAll = cell(length(times), 1);
gAll = cell(length(times), 1);



%I_x = blkdiag(eye(n_p),eye(n_cAll{1}));
I_x = I_xAll{2};
FAll{1} = blkdiag(I_x,D);

price_Err = cell(length(times), 1);
norm_Err = zeros(length(times), 1);
PAll = cell(length(times), 1);
PpredictAll = cell(length(times), 1);
PTAll = cell(length(times), 1);

mu_xAll = cell(length(times), 1);

new_price_Err = cell(length(times), K);

%ansätt en lösning för ALLA tidsprioder


%load x_start3000.mat xAll
for t = 1:length(times(1:TT))
    
    %Använder givna terminsräntekurvor för att estimera states x_p, resten
    %sätts till 0. 
    n_u = size(D,1);
    x_p = Eeig' * fH(t,1:3661)';
    x_u = zeros(n_u,1); 
    x_c = zeros(n_cAll{t},1); 
    x_s = [x_p; x_c];
    x = [x_s; x_u];
    z = priceAll{t};
    I_z = I_zAll{t};
    %I_z* z- g +  + vAll{t};

    g = [];
    o_inst = oAll{t};
    for j=1:length(o_inst)
        o_inst = oAll{t};
        o_tmp = o_inst{j};
        dTtmp = year_fracAll{t};
        deltaT = dTtmp{j};    
        o_x =  o_tmp* x_s;
        g = [g; (exp(o_x(1)) - exp(o_x(end))) / (exp(o_x(2:end))' * deltaT)];  
    end

    %if size(z,1) ~= 27
    %    disp(z)
    %end

    vAll{t} = z-g -I_z * x_u;

    n_z = size(priceAll{t},1);
    
    w = zeros(size(x));

    xAll{t} = x;
    wAll{t} = w;
    %vAll{t} = v;

    g = [];
    o_inst = oAll{t};
    for j=1:length(o_inst)
        o_inst = oAll{t};
        o_tmp = o_inst{j};
        dTtmp = year_fracAll{t};
        deltaT = dTtmp{j};
        
        o_x =  o_tmp* x_s;
        g = [g; (exp(o_x(1)) - exp(o_x(end))) / (exp(o_x(2:end))' * deltaT)];  
    end

    price_Err{t} = abs(g-z);

    %x_p = ones(6,1)/10;
    %x_p(1) = 0.8;

    %f = Eeig * x_p;

%    test = I_z* z- g - I_z * x_u -vAll{t};


    %Ansätt värde för P (kovariansmatris för states x)
    
    test = 1.47953530424543e-09;

    tmp = test*diag(ones(size(x)))/10;
    tmp(1,1) = ef(1);
    tmp(2,2) = ef(2);
    tmp(3,3) = ef(3);
    tmp(4,4) = ef(4);
    tmp(5,5) = ef(5);
    tmp(6,6) = ef(6);
    
    tmp(1,1) = 1.15382806115773e-07;
    tmp(2,2) = 7.96098112382795e-08;
    tmp(3,3) = 1.77603452419612e-08;
    tmp(4,4) = 9.88303931125785e-09;
    tmp(5,5) = 4.05152569797899e-09;
    tmp(6,6) = 1.47953530424543e-09;
    %tmp(7,7) = tmp(1,1);
    %x_u = x(end-26:end);
    tmp2 = test*diag(ones(size(x_u)))/100;

    tmp(end-n_u+1:end, end-n_u+1:end) = tmp2;

    PTAll{t} = tmp*100000;
end

%ansätter kovariansmatriser för w och v.
w= wAll{1};
v = vAll{1};
%Qw = diag(ones(size(w)))/10;

tmp = diag(ones(size(w)));


%tmp(1,1) = ef(1);
%tmp(2,2) = ef(2);
%tmp(3,3) = ef(3);
%tmp(4,4) = ef(4);
%tmp(5,5) = ef(5);
%tmp(6,6) = ef(6);

Qw = PTAll{1};

Rv = test*diag(ones(size(v)));


x0 = xAll{1};


tmp = 0;
for t = 1:length(times(1:TT))
    tmp = tmp + mean(abs(price_Err{t}));
end
start_fel = tmp/TT;

c_sumAll = zeros(K,1);
for k = 1:K
    %sätter mu_x till F*x_{t-1}
    F = FAll{1};
    mu_xAll{1} = F*x0;

    for t = 2:length(times(1:TT))
        x = xAll{t-1};
        F = FAll{t};
        mu_x = F*x;
        mu_xAll{t} = mu_x;
      
    end
    % beräknar H och g givet nuvarande lösning samt medelprisfel för
    % nuvarande lösning
    for t = length(times(1:TT)):-1:1
        x = xAll{t};
        if t == 1
            x_prior = x0;
        else
            x_prior = xAll{t-1};
        end
        x_s = x(1:end-n_u);
        x_u = x(end-n_u+1:end);
        z = priceAll{t};
        I_z = I_zAll{t};
        grad_g = [];
        g = [];
     
        o_inst = oAll{t};
        for j=1:length(o_inst)
            
            o_tmp = o_inst{j};
            dTtmp = year_fracAll{t};
            deltaT = dTtmp{j};
            
            o_x =  o_tmp* x_s;
            g = [g; (exp(o_x(1)) - exp(o_x(end))) / (exp(o_x(2:end))' * deltaT)];  
            %grad_g = [grad_g ; (o_tmp(1) * exp(o_x(1)) - o_tmp(end)* exp(o_x(end))) / (exp(o_x(2:end))' * deltaT) - (exp(o_x(1)) - exp(o_x(end)) ) * exp(o_x(2:end))' * diag(deltaT) *o_tmp(2:end,:) / (exp(o_x(2:end))' * deltaT)^2;];
            % Numerator and denominator        
        end
        H = [grad_g I_z];

        %calculate gradient numerically

        

        h = 10^-12;
        num_grad = [];
        
        for i = 1:length(x_s)
            inner_grad = [];
            for j=1:length(o_inst)
                o_tmp = o_inst{j};
                dTtmp = year_fracAll{t};
                deltaT = dTtmp{j};
                h_vec = zeros(size(x_s));
                h_vec(i) = h;
                o_x =  o_tmp* (x_s+h_vec);
    
                num_grad1 = (exp(o_x(1)) - exp(o_x(end))) / (exp(o_x(2:end))' * deltaT);
                o_x =  o_tmp* (x_s-h_vec);
                num_grad2 = (exp(o_x(1)) - exp(o_x(end))) / (exp(o_x(2:end))' * deltaT);
    
                inner_grad = [inner_grad; (num_grad1 - num_grad2 )/2/h];
            end
            num_grad = [num_grad, inner_grad];
        end

        H = [num_grad I_z];

        HAll{t} = H;
        gAll{t} = g;
        tmp = abs(g-z);

        
        
    end

    Q_post = 0;
    q_post = 0;
    d_post = 0;
    
    %räknar ut parametrar som beror av kovariansmatriser
    c_sum = 0; %c_sum används för att se vad totala målfunktionsvärdet av det icke-approximerade problemet var innan iteration görs. Dvs det ska alltid öka
    for t = length(times(1:TT)):-1:1
        F = FAll{t};
        H = HAll{t};
        z = priceAll{t};
        g = gAll{t};
        x = xAll{t};
        I_z = I_zAll{t};
        if t == 1
            x_prior = x0;
        else
            x_prior = xAll{t-1};
        end
        x_s = x(1:end-n_u);
        x_u = x(end-n_u+1:end);
        v = vAll{t};
        w = wAll{t};
        
    
        mu_w = zeros(size(w));
        mu_v = zeros(size(v));
        P = PTAll{t};

        n = size(x,1);
        nz = size(v,1);
        nQ = size(Qw,1);
        if n > nQ
            %tmp = [mu_x; mu_x(end-n+57+1:end)];
            sigma_w = blkdiag(Qw, Qw(end-n+nQ+1:end,end-n+nQ+1:end));
        else
            %tmp = mu_x(1:n);
            sigma_w = Qw(1:n, 1:n); 
        end

        if nz > n_u
            sigma_v = blkdiag(Rv, Rv(end-nz+n_u+1:end,end-nz+n_u+1:end));         
        else
            sigma_v = Rv(1:nz, 1:nz);
        end

        mu_x = mu_xAll{t};
        [c_w , a, A] = loglikelihood(w, mu_w, sigma_w, 'normal', 0);
        [c_v , b, B] = loglikelihood(v, mu_v, sigma_v, 'normal', 0);
        [c_x , e, E] = loglikelihood(x, mu_x, P, 'normal', 0);

        c = c_x + c_v + c_w;
        
        c_sum = c_sum + c;
        delta_v = v - z + g + I_z * x_u ; %alla vektorer har olika dimensioner?
        delta_w = F * x_prior + w - x;
            
        %delta_v = -delta_v;

        %eq 4.60
        c_overline = c - a'*delta_w + 1/2 * delta_w' * A * delta_w - b'*delta_v + 1/2 * delta_v' * B * delta_v + d_post;
        
        a_overline = a - H'  * b  + e - A * delta_w  + H' * B * delta_v + q_post;  

        r_overline = -F'*a + F' * A * delta_w;

        A_overline = A + E + H' * B * H + Q_post;

        R_overline = F' * A * F;

        P_overline = F' * A;

        d = c_overline - 1/2 * a_overline' * (A_overline \ a_overline);
        q = r_overline + P_overline * (A_overline \ a_overline);
        Q = R_overline - P_overline * (A_overline \ P_overline');

        delta_vAll{t} = delta_v;
        delta_wAll{t} = delta_w;
        
        P_overlineAll{t} = P_overline;
        A_overlineAll{t} = A_overline;
        a_overlineAll{t} = a_overline;





        Q_post = Q;
        q_post = q;
        d_post = d;
    end
    c_sumAll(k) = c_sum;
    
    P = PTAll{1};
    mu_x = mu_xAll{1};
   
    [c_x , e, E] = loglikelihood(x0, mu_x, P, 'normal', 0);
    A = E + Q;
    a = e + q;

    
    
    x_prior_update =  -A \ a;
    x0 = x0 +  x_prior_update;
    
    f_0 = c_x + e' * x_prior_update + x_prior_update' * E* x_prior_update + d + q' * x_prior_update + 1/2 * x_prior_update' * Q * x_prior_update;
    
    if k~=1
       if f_old - f_0 > 0
            k
            break
        end
    %f_old - f_0;
    end

    f_old = f_0;
    
    for t = 1:length(times(1:TT))
        A_overline = A_overlineAll{t};
        P_overline = P_overlineAll{t};
        a_overline = a_overlineAll{t};
        F = FAll{t};
        H = HAll{t};
        delta_w = delta_wAll{t};
        delta_v = delta_vAll{t};
        
        x_update = A_overline \ P_overline' * x_prior_update - A_overline \ a_overline;


        %x_update = step_size * x_update;

        w_update = x_update - F * x_prior_update - delta_w;
        v_update = - H * x_update - delta_v;


        if size(xAll{t},1) ~= size(x_update,1)
            disp("fel på x")
        end
        if size(wAll{t},1) ~= size(w_update,1)
            disp("fel på w")
        end
        if size(vAll{t},1) ~= size(v_update,1)
            disp("fel på v")
        end
        v_old = vAll{t};
        x_old = xAll{t};
        x_u_old =x_old(end-n_u+1:end);
        x = xAll{t} + x_update;
        w = wAll{t} +w_update;
        v = vAll{t} +v_update;
        x_prior_update = x_update;
        xAll{t} = x;
        wAll{t} = w;
        vAll{t} = v;

        tmp = CAll{t};
        if  ~isnan(tmp)
            C = CAll{t};
            n = size(C,2);
            C = [C;ones(1,n)];
        end
       
        x = xAll{t};
        x_s = x(1:end-n_u);
        
        g = [];
        o_inst = oAll{t};
        for j=1:length(o_inst)
            o_inst = oAll{t};
            o_tmp = o_inst{j};
            dTtmp = year_fracAll{t};
            deltaT = dTtmp{j};
            
            o_x =  o_tmp* x_s;
            g = [g; (exp(o_x(1)) - exp(o_x(end))) / (exp(o_x(2:end))' * deltaT)];  
        end
    
        z = priceAll{t};
    
    
        new_price_Err{t,k} = z-g;
        
       
    end

    tmp = 0;
    for t = 1:length(times(1:TT))
        tmp = tmp + mean(abs(new_price_Err{t,k}));
    end

    tmp/TT

end



C= CAll{1};
fAll = zeros(size(C,1)+1, TT);

kalman_price = cell(1,TT);

load Copy_of_OISdata.mat

n_uOld = n_u;

for t=1:length(times(1:TT))
    tmp = CAll{t};
    n_u = size(D,1);

    if  ~isnan(tmp)
        C = CAll{t};
        n = size(C,2);
        C = [C;ones(1,n)];
    end

    z = priceAll{t};
    I_z = I_zAll{t};
    
    x = xAll{t};%+wAll{t};
    
    x_s = x(1:end-n_uOld);
    x_u = x(end-n_uOld+1:end);

    f = [Eeig C] * x_s;

    fAll(:,t) = f;



    

    %x_s(7) = x_s(7) + 0.01;

    %f = [Eeig C] * x_s;
    %figure(4)
    %plot(f);
    
    g = [];
    o_inst = oAll{t};
    for j=1:length(o_inst)
        o_inst = oAll{t};
        o_tmp = o_inst{j};
        dTtmp = year_fracAll{t};
        deltaT = dTtmp{j};
        
        o_x =  o_tmp* x_s;
        g = [g; (exp(o_x(1)) - exp(o_x(end))) / (exp(o_x(2:end))' * deltaT)];  
    end

    kalman_price{t} = g;

    %x_p = ones(6,1)/10;
    %x_p(1) = 0.8;

    %f = Eeig * x_p;

    %test = z- g - I_z * x_u -vAll{t};

    %mean_price_Err(t,K+1) = mean(test, "all");
    
end
%avrundnings_fel = mean(mean_price_Err(:,K+1))
tmp1 = times(1:TT);
tmp2 = (1:size(fAll,1))/365;
tmp3 = fAll(1:end,:);

figure(3)
plot3DCurve(tmp1,tmp2 ,tmp3 );

figure(4)
tmp3 = fH(1:TT,:)';
plot3DCurve(tmp1,tmp2 ,tmp3 );

error = zeros(length(times), 27);

for t=1:length(times(1:TT))

    tmp = kalman_price{t};
    cnt = 0;
    prices = priceAll{t};
    for i = 1:length(prices)
        if i == remove_assetNo
            usedInstr(t,i) = 1;
        end

        cnt = cnt + usedInstr(t,i);
        if usedInstr(t,i)
            error(t,i) = tmp(cnt) - prices(cnt);
        end


    end

end

mean(abs(error(:,remove_assetNo)))