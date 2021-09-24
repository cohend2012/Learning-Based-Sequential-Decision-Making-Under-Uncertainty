%Markov HW
%Dan S. Cohen
%DATE: 9/6/2021
% Assigment Number 1 
%% Inputs

%clear all; clc;

n = input('Number of sates? ') % Number of sates = n
P = zeros(n,n)
PL = zeros(n,n)
% Transistion probability matrix = P
% Inital Distibution P0
P0 = zeros(n,n)
P0(1,1)= 1

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
Reward = zeros(n,1)

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
    
    weight = zeros(i,1)
    for ii = 1:i-1
        
        weight(end-ii)=ii;
        
    end
        
        
    Reward(i) =   -(ii)*m+dot(P(i,1:i),weight)*r-p*(((M-1)/2)-ii);  
    
    
    
end
Expected_Profit_per_stage =0;
Expected_Profit_per_stage_saved_per=zeros(1,k);
for t=1:k
    
    Expected_Profit_per_stage=Expected_Profit_per_stage+P0*((P^(t-1))*Reward);
    
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






%S = zeros(1,length(k)); % stock level
D = zeros(1,length(k)); % demand
U = zeros(1,length(k)); % policty to restock


S(1) = 0;
S(2:end)=S(2:end)-1

for stage = 1:length(k)
    
    %randi(M,1,1)
    
    maint_cost = m*S(stage);
    
    D(stage) = round(rand*M);
    disp('demand')
    disp(D(stage))
    % rolling avg control pi

    %S_avg(stage) = sum(S(1:stage))/stage
     
    %maint cost
    
    
    %U(stage) = abs(M-S_avg(stage)) % need to work out policy
    if S(stage) <= 0
        U(stage) = M;
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
                % halfcost 
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
title('Plot of P_K(BAR) and P_K(HAT)') 

%%
% Part 2
Transition = {}
count = 0

total_d = 0;
d = 1/M;
start_number = 1
for action = 1:M
    Transition{action} = zeros(M);
    total_d = 0;
    
    for i = 1:M-count
        
        Transition{action}(i,1)= start_number-total_d
        total_d = total_d + d;

           
      
    end 
 
    start_number =  start_number-d     ;  
    
    count = count +1 ;
    
 
    
end

count=0
col_count = 3
start_number_count = 0
for action = 1:M
    
   
     for i = 1:M-count
     
          j = start_number_count:col_count
              
                Transition{action}(i,2:j(i)+1)=1/M
          
          
     end 
        
    start_number_count = start_number_count+1  ;
    count = count +1 ;
   
  
 
end



Reward_From_Action = {}

%Reward(1)=-3*c+((M-1)/2)*r;
%Reward(n)=-(M-1)*m+((M-1)/2)*r;


for act = 1:M
    
    weight = zeros(i,1)
    for ii = 1:i-1
        
        weight(end-ii)=ii;
        
    end
    
    for j = 1:M
    
        for i= 1:M
            
            if (i+a-j)> 0
                rew_units = (i+a-j);
            end 
            if (a+i)< i-M
                Dk = M-(a+i)-1 % fix this 
                
                
            end
        
            Reward_From_Action{act}(i,j) =   -(i)*m-(act)*c+rew_units*r-p*(((M-1)/2)-ii);
        
        end  
        
    end 
      
    
    
    
end


V0 = 0

V = zeros(1,length(k));

V(end)=V0


k = length(k)-1;

policy = zeros(1,length(k));
V_new = 0 
V_old = 0
curr_val = 0
while k >0

    for act=1:M
    
    
        for i=1:M
            curr_val =0 
            for j=1:M
                                    
               curr_val = curr_val + Transition{act}(i,j)*V(k)
            
            end 
            V(i)=Reward_From_Action{act}(i,j) + curr_val
            if V_new > V_old
            policy(k) = act
            
       
        end
        
        
        end
    
    end 
    
    k = k - 1

end






