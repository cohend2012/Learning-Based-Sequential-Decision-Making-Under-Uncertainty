%Markov HW
%Dan S. Cohen
%DATE: 9/6/2021
% Assigment Number 1 
%% Inputs

clear all; clc;

n = input('Number of sates? ') % Number of sates = n
P = zeros(n,n);
PL = zeros(n,n);
% Transistion probability matrix = P
% Inital Distibution P0
P0 = zeros(n,n);
P0(1,1)= 1;

% Number of stages 
k = input('Number of stages? ') % Number of sages = k

%k = [1,2,3,4,5,6,7,8,9,10,11,12];% stages


% Calc Profit 
m = 0.5; % mait cost
c = 1; % cost of each item
r = 2; % rev of each unit
p = 1; % penalty
M = n; % max units
%k = linspace(1, k, k)
Reward = zeros(n,1);

%%
total_d = 1/M;
d = 1/M;

P(1,:) = d;
P(n,:) = d;
first_e = true;


    
for j=2:n-1
    P(j,1)=1-total_d;
   
    total_d = total_d + d;
        
end

count = 2;
for i=2:n-1
    
    for col=2:count
        PL(i,col)= d;
        
    end
    count = count +1;
    
    
end

P = P + PL;


Reward(1)=-3*c+((M-1)/2)*r;
Reward(n)=-(M-1)*m+((M-1)/2)*r;

for i=2:n-1
    
    weight = zeros(i,1);
    for ii = 1:i-1
        
        weight(end-ii)=ii;
        
    end
        
        
    Reward(i) =   -(ii)*m+dot(P(i,1:i),weight)*r-p*(((M-1)/2)-ii);  
    
    
    
end
Expected_Profit_per_stage =0;
Expected_Profit_per_stage_saved_per=zeros(1,k);
% Part 1 starts
for t=1:k
    
    
    
    % belive I need to add in the half priced end
    
    if t==k
        
        Expected_Profit_per_stage = Expected_Profit_per_stage+(P0*((P^(t-1))*Reward))/2;
        
        
    else 
        Expected_Profit_per_stage=Expected_Profit_per_stage+P0*((P^(t-1))*Reward);
        
        
    end 
    
    Expected_Profit_per_stage_saved_per(t)=Expected_Profit_per_stage(1)/t;
    
end 

Expected_Profit_per_stage=Expected_Profit_per_stage/k;

%%


rng(0);
for i =1:k
    
    U_vector(i)=rand;
    
    G(i)=-log(-log(U_vector(i)));
    
end 


Uniform_var = rand;
G(1) = -log(-log(Uniform_var));


k_m1 = 1;
k = linspace(1, k, k);
S = zeros(1,length(k)); % stock level
for stage =1:length(k)
    
    
    
    U_vector=rand(1,n);
    
    G = -log(-log(U_vector));
    
    
    [maxV,index] = max(G + log(P(k_m1,1:n))) ;
    S(stage) = index;
    k_m1 = S(stage);
    
    
    
end 




% for states first one, layz 

%S = zeros(1,length(k)); % stock level
D = zeros(1,length(k)); % demand
U = zeros(1,length(k)); % policty to restock


S(1) = 0;
S(2:end)=S(2:end)-1;

for stage = 1:length(k)
    
    %randi(M,1,1)
    
    maint_cost = m*S(stage);
    
    D(stage) = randi([0,M-1],1);
    disp('demand')
    disp(D(stage))
    % rolling avg control pi

    %S_avg(stage) = sum(S(1:stage))/stage
     
    %maint cost
    
    
    %U(stage) = abs(M-S_avg(stage)) % need to work out policy
    if S(stage) <= 0
        U(stage) = M-1;
        Profit(stage) =   -(c*U(stage))+r*D(stage);
        disp('restocking all')
        
    elseif S(stage) > 0
        U(stage) = 0;
        disp('not buying any more stock')
            

   
        if(D(stage)> S(stage)+U(stage))
            disp('demand biger then stock')
            Profit(stage) = -(maint_cost+c*U(stage))+r*(S(stage)+U(stage))-p*(D(stage)-S(stage)-U(stage));
        
    
        end 
    
        if(D(stage)<= S(stage)+U(stage))
            disp('demand less then stock')
            Profit(stage) =   -(maint_cost+c*U(stage))+r*D(stage);
        
            if (stage ==length(k))
            
                disp('left over')
                % Halfcost 
                Profit(stage) =Profit(stage)+ 0.5*(U(stage)-D(stage))
            
            end 
       
        
        end

    end 

    disp('total supply')
    
    
       

        
    
    %S(stage+1)=S(stage)+U(stage)-D(stage)
    
    %if(S(stage+1)<= 0)
        
        %S(stage+1) = 0;
        
    %end 
    
    %Profit_per_stage(stage)= Profit_per_stage(stage) + Profit(stage)/stage
    
    
end

disp('Profit per stage')
disp(sum(Profit)/length(k))

Profit_per_stage = zeros(1,length(k));

for key =1:length(k)
    
       temp= Profit(1:key);
       
       Profit_per_stage(key) = sum(temp)/key;
    
    
end 





one = size(P);


w = [eye(one(1))-P,ones(one(1),1)];

% Avg reward per stage 

PIE = [ones(1,one(1))]*[w*w.']^-1;

PerStageProfit = PIE*Reward

 K =[1:1:length(k)]
 

plot(K,PerStageProfit*ones(1,length(k)))
hold on 
plot(K,Expected_Profit_per_stage_saved_per)
hold on 
plot(K,Profit_per_stage)
title(' Question 4 Plot of P_K(BAR) and P_K(HAT)') 

%%
% Part 2
Transition = {};
count = 0;

total_d = 0;
d = 1/M;
start_number = 1;
% for action = 1:M
%     Transition{action} = zeros(M);
%     total_d = 0;
%     
%     for i = 1:M-count
%         
%         Transition{action}(i,1)= start_number-total_d
%         total_d = total_d + d;
% 
%            
%       
%     end 
%  
%     start_number =  start_number-d     ;  
%     
%     count = count +1 ;
%     
%  
%     
% end
% 
% count=0;
% col_count = M-1;
% start_number_count = 0;
% for action = 1:M
%     
%    
%      for i = 1:M-count
%      
%           j = start_number_count:col_count;
%               
%                 Transition{action}(i,2:j(i)+1)=1/M;
%           
%           
%      end 
%         
%     start_number_count = start_number_count+1  ;
%     count = count +1 ;
%    
%   
%  
% end

Transition  = {};



count=0;
for j=1:1
    
    Transition{j} = zeros(M);
   
    for i = 1:M
        total_d = 0;
        for a = 1:M-count
        
            Transition{j}(a,i) = start_number-total_d
            total_d = total_d + d;    
        
        end 
        start_number =  start_number-d
        count = count + 1 
    end 
    
         
    
end 


for j=2:M
    
    Transition{j} = zeros(M);
   
    for i = 1:M
        total_d = 0;
        for a = 1:M
            if  (i+a)-1-1<=M-1 && (j-1 <= i + a -1 -1)
                Transition{j}(a,i) = d ; 

            end 
              
        
        end 
        
    end 
    
         
    
end 


Temp  = Transition;

for a=1:M
    
    
    for i=1:M
        
        for j=1:M
            
            Transition{a}(i,j) = Temp{j}(a,i)
            
            
        end 
        
        
        
    end 
    
    
    
    
   
    
    
end 






















Reward_From_Action = {};

%Reward(1)=-3*c+((M-1)/2)*r;
%Reward(n)=-(M-1)*m+((M-1)/2)*r;


for i = 1:M
    
    weight = zeros(i,1);
    for ii = 1:i-1
        
        weight(end-ii)=ii;
        
    end
    
    for demand = 1:M
    
        for act= 1:M
            
            j = i+a-demand;
            
            
            
            if(i+act)-1-1>M-1 ||(i+act)-1-1<j-1 % Is it even posible  grater or equal?
                Reward_From_Action{i}(act,demand)= 0;
            
            else
                             
                if demand > i+act % should the system get reward
                    
                    
                    rew_units = (i+act-j-1); % reward 
        
                   Reward_From_Action{i}(act,demand) = -(i)*m-(act)*c+rew_units*r-p*(demand-i-act);
        
                else 
                    rew_units = (demand-1); % reward 
                   Reward_From_Action{i}(act,demand) = -(i)*m-(act)*c+rew_units*r;
                end
                
             
                
                
            end
        end
    end
end



for i = 1:M 
    
    for act = 1:M 
        curr_profit_val = 0 ;
        for demand = 1:M 
            
            curr_profit_val = curr_profit_val+ Reward_From_Action{i}(act,demand);
            
            
        end 
        avg_profit(i,act) = curr_profit_val/M;
        
        
        
    end 
    
    
    
    
end 






V_new = ones(M,1)*0;
V_new(M) = 0;
V_old = 0;
curr_val = 0;
        
% value interation per stage
val_iter_profit = 0 
V0 =0;

%V(end)=V0;
% Value interation, at each k in time slot 

k = length(k);
policy_over_time = zeros(1,k);
V = ones(M,k)*-inf;
termal_profit = [0:M-1]*(r/2);
V(:,end) = termal_profit;
while k >0 
    for i=1:M          
        for act=1:M
            curr_val =0 ;
            curr_val = avg_profit(i,act)+Transition{act}(i,:)*V(:,k);            
            if k-1<=0            
                break    
            end 
            %V(i,k-1) = max(V(i,k-1),curr_val)
            if V(i,k-1) < curr_val
                V(i,k-1) = curr_val;
                recored_actions(k) = act;
                recored_index(k)   = i;

            end 
        end 
        
    end 
    
    k = k - 1;
    
end 


%k = length(K);

total_profit = 0 ;
for k=1:length(K)
    
    if k == 1
        
        Best_Value= V(1,k);
        
        Best_Index = 1;
    else 
        
        [Best_Value,Best_Index]=max(V(:,k));    
    end 
    policy_over_k(k) = Best_Index;

end 

expected_profit_from_empy_warehouse = V(1,1);



%tot_val_iter_profit_perk = val_iter_profit/length(K)




V0 =0;

%V = zeros(1,M);

%V(end)=V0;

% Policy iteration static per state
k = length(k);

policy = ones(1,M);
V_new = ones(M,1)*0;
h_new = ones(M,1)*0;
V_new(M) = 0;
V_old = 0;
curr_val = 0;

% policy interations % finds best policy at each stock level
time_start = tic;


for i=1:M
    
    
    for act=1:M
        curr_val =0 ;
        for j=1:M
                                    
            curr_val = curr_val + Reward_From_Action{i}(act,j)+Transition{j}(act,i)*V(j);
            
         end 
         %curr_val= + curr_val;
            
         %V_new(i) = max(V_new(i),curr_val);
          
            
         if V_new(i) <= curr_val
                
             policy(i) = act;
                
             V_new(i) = max(V_new(i),curr_val);
        
          
         end 
            
       
     end
     
%     max_diffrace = 0;
%     
%     for states = 1:M
%         max_diffrace = max(max_diffrace,V_new(states));
%         
%     end
        

     %V = V_new;
%      end
%      if(max_diffrace<0.01)
%          break
end 
total_time_run = toc(time_start);     

figure(2)
plot(1:M,policy)
hold on
title('States vs Policy') 
% policy inter with ac/s
policy = ones(1,M);
h = zeros(M,1);
h_new = zeros(M,1);

V = 1*-inf;
for pol_itter = 1:1000

    for i=1:M
        act = policy(i);
        curr_val =0 ;
 
        curr_val = avg_profit(i,act)
                                    
        %curr_val = curr_val + Reward_From_Action{i}(act,j)+Transition{j}(act,i)*V(j);
        %curr_val= + curr_val;
        %V_new(i) = max(V_new(i),curr_val);
            
        if V <= curr_val   
            V = curr_val
        end 

    end

     % Find h_tilida
    for iter = 1:10
        for i=1:M
            act = policy(i);
            curr_val =0 ;
             
            curr_val = avg_profit(i,act)+Transition{act}(i,:)*h-V;      
            curr_val = curr_val + V                       
            saved_trans = Transition{act}(i,:)
            saved_reward(i) = avg_profit(i,act);
            
            %h(i) = curr_val;
            %curr_val= + curr_val;
            if h(i) <= curr_val
                
                best_saved_reward = saved_reward;
                best_saved_trans  = saved_trans;
                
                h(i) = curr_val;
                
            end 

        end

    end 

    %x = [h;V]
    I_tilda = eye(M);
    I_tilda(1,:) = [];
    %h_tilda = I_tilda*h;
    %y = [h_tilda;V];

    W = [eye(M)-best_saved_trans,ones(M,1)]*[I_tilda',zeros(M,1);zeros(1,M-1),1]; % 5x4 ;
    y = inv(W)*best_saved_reward';
    h_tilda = y(1:M);

    curr_policy = policy ;
    for i=1:M
            best_val = 0;
            
            for act = 1:M
                
                curr_val = 0;
               
                curr_val = avg_profit(i,act)+Transition{act}(i,:)*h_tilda;
  
                 if best_val < curr_val
                 
                    best_val = curr_val;
                    
                    policy(i) = act;

                 end
            end
        % Do we need to stop
    end 
        % is the poly stable

        if abs(policy-curr_policy)==0        
        
            break
            disp('ready')

        end 

end 


disp('Best Policy is...')
disp(policy)










% Part 4


S = zeros(1,length(K)); % stock level
%D = zeros(1,length(k)); % demand
U = zeros(1,length(K)); % policty to restock


%S(1) = 0;
%S(2:end)=S(2:end)-1;
planed_pol = [linspace(M, 1, M)];

% optimal system

%policy(16:20) = 1;
for stage = 1:length(K)
    
    %randi(M,1,1)
    
    maint_cost = m*S(stage);
    
    %D(stage) = round(rand*M);

    % rolling avg control pi

    %S_avg(stage) = sum(S(1:stage))/stage
     
    %maint cost
    
    U(stage) = policy(S(stage)+1)-1 % poly control
    
    S(stage+1) = (U(stage)-D(stage))
    
    if(S(stage+1)<0)
        
       S(stage+1)=0;
       
    end
    
    if(S(stage)<0)
        
       S(stage)=0;
       
    end
    
    
    
    
    
    disp('demand')
    %disp(D(stage))
    
    %U(stage) = abs(M-S_avg(stage)) % need to work out policy
    if S(stage) <= 0
        %U(stage) = M;
        Profit_opt(stage) =   -(c*U(stage))+r*D(stage);
        disp('restocking all')
        
    elseif S(stage) > 0
        %U(stage) = 0;
        disp('not buying any more stock')
            

   
        if(D(stage)> S(stage)+U(stage))
            disp('demand biger then stock')
            Profit_opt(stage) = -(maint_cost+c*U(stage))+r*(S(stage)+U(stage))-p*(D(stage)-S(stage)-U(stage));
        
    
        end 
    
        if(D(stage)<= S(stage)+U(stage))
            disp('demand less then stock')
            Profit_opt(stage) =   -(maint_cost+c*U(stage))+r*D(stage);
        
            if (stage ==length(k))
            
                disp('left over')
                % Halfcost 
                Profit_opt(stage) =Profit_opt(stage)+ 0.5*(U(stage)-D(stage))
            
            end 
       
        
        end

    end 

    disp('total supply')
    
    
       

        
    
    %S(stage+1)=S(stage)+U(stage)-D(stage)
    
    %if(S(stage+1)<= 0)
        
        %S(stage+1) = 0;
        
    %end 
    
    %Profit_per_stage(stage)= Profit_per_stage(stage) + Profit(stage)/stage
    
    
end



disp('Profit per stage')
disp(sum(Profit_opt)/length(K))

Profit_per_stage_opt = zeros(1,length(K));

for key =1:length(K)
    
       temp= Profit_opt(1:key);
       
       Profit_per_stage_opt(key) = sum(temp)/key;
    
    
end 

% lazy policy with same demand

S = zeros(1,length(K)); % stock level
%D = zeros(1,length(k)); % demand
U = zeros(1,length(K)); % policty to restock

%S(1) = 0;
%S(2:end)=S(2:end)-1;


%policy(16:20) = 1;

pol=[zeros(1,M)];
pol(1)=M-1;
for stage = 1:length(K)
    
    %randi(M,1,1)
    
    maint_cost = m*S(stage);
    
    %D(stage) = round(rand*M);

    % rolling avg control pi

    %S_avg(stage) = sum(S(1:stage))/stage
     
    %maint cost
    
    U(stage) = pol(S(stage)+1);
    
    S(stage+1) = (U(stage)-D(stage));
   
    
   if(S(stage+1)<0)
        
       S(stage+1)=0;
       
    end
    
    if(S(stage)<0)
        
       S(stage)=0;
       
    end
    
    disp('demand')
    %disp(D(stage))
    
    %U(stage) = abs(M-S_avg(stage)) % need to work out policy
    if S(stage) <= 0
        %U(stage) = M;
        Profit_lazy(stage) =   -(c*U(stage))+r*D(stage);
        disp('restocking all')
        
    elseif S(stage) > 0
        %U(stage) = 0;
        disp('not buying any more stock')
            

   
        if(D(stage)> S(stage)+U(stage))
            disp('demand biger then stock')
            Profit_lazy(stage) = -(maint_cost+c*U(stage))+r*(S(stage)+U(stage))-p*(D(stage)-S(stage)-U(stage));
        
    
        end 
    
        if(D(stage)<= S(stage)+U(stage))
            disp('demand less then stock')
            Profit_lazy(stage) =   -(maint_cost+c*U(stage))+r*D(stage);
        
            if (stage ==length(k))
            
                disp('left over')
                % Halfcost 
                Profit_lazy(stage) =Profit_lazy(stage)+ 0.5*(U(stage)-D(stage))
            
            end 
       
        
        end

    end 

    disp('total supply')
    
    
       

        
    
    %S(stage+1)=S(stage)+U(stage)-D(stage)
    
    %if(S(stage+1)<= 0)
        
        %S(stage+1) = 0;
        
    %end 
    
    %Profit_per_stage(stage)= Profit_per_stage(stage) + Profit(stage)/stage
    
    
end



disp('Profit per stage lazy')
disp(sum(Profit_lazy)/length(K))

%Profit_per_stage_opt = zeros(1,length(K)); 

for key =1:length(K)
    
       temp= Profit_lazy(1:key);
       
       Profit_per_stage_lazy(key) = sum(temp)/key;
    
    
end 




figure(3)
plot(1:length(K),Profit_opt)
hold on
plot(1:length(K),Profit_lazy)
title('Optimal vs lazy invotory mgmt') 
xlabel('stages k')
ylabel('Units '); 
legend('Profit_opt','Profit');

disp('percent profit better')
disp((sum(Profit_opt)/sum(Profit_lazy))*100)




figure(4)
plot(1:length(K),Profit_opt)
hold on
plot(1:length(K),Profit_per_stage_opt)
plot(1:length(K),Profit_per_stage_lazy)
title('Optimal vs lazy invotory mgmt') 
xlabel('stages k')
ylabel('Units '); 
legend('Profit_opt','Profit_per_stage_opt','Profit_per_stage_lazy');



% page 64 calculation for avg cost per stage
curr_val_V_mu = zeros(1,M);
for i=1:M 
    
    act = policy(i);
    
    for j=1:M
        curr_val_V_mu(i) = curr_val_V_mu(i) + Reward_From_Action{i}(act,j);
        
    end
    
    
    
end 


disp('for avg cost per stage')
disp(curr_val_V_mu)






           %h_new(i) = curr_val;
            
           %h_new(i) = max(h_new(i),curr_val);
%             if V_new(i) <= curr_val
%                 
%                 policy(i) = act;
%                 
%                 V_new(i) = max(V_new(i),curr_val);
%           
%             end 


    
%     max_diffrace = 0;
    
%     for states = 1:M
%         max_diffrace = max(max_diffrace,abs(h(states)-h_new(states)));
%         
%     end

        %h = h_new;

%         if(max_diffrace<0.1)
%             break
%          
%         end 

%     
%      
%     max_diffrace = 0;
%     
%     for states = 1:M
%         max_diffrace = max(max_diffrace,abs(V(states)-V_new(states)));
%         
%     end
%         
% 
%         V = V_new;
%        
%      
%      
%         if(max_diffrace<0.1)
%             break
%          
%         end 
%     
%     
    
