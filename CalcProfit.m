%Markov HW
%Dan S. Cohen
%DATE: 9/6/2021
% Assigment Number 1 
%% Inputs
n = input('Number of sates? ') % Number of sates = n
% Transistion probability matrix = P
% Inital Distibution P0
% Number of stages 
k = input('Number of stages? ') % Number of sages = k

%k = [1,2,3,4,5,6,7,8,9,10,11,12];% stages


% Calc Profit 
m = 100;
c = 10; % cost of each item
r = 1000; % rev of each unit
p = 1; % penalty
M = 100; % max units
k = linspace(1, k, k)

%%


S = zeros(1,length(k));
D = zeros(1,length(k));
for stage = 1:length(k)
    
    S(stage) = round(rand*M);
    D(stage) = round(rand*M);
    
end

U = zeros(1,length(k));


% rolling avg control pi

for stage = 1:length(k)
    
    S_avg(stage) = sum(S(1:stage))/stage
        

    
    U(stage) = abs(M-S_avg(stage))
    
end

for stage = 1:length(k)
    
   
    if(D(stage)> S(stage)+U(stage))
        Profit(stage) = -(m*S(stage)+c*U(stage))+r*(S(stage)+U(stage))-p*(D(stage)-S(stage)-U(stage));
        
    
    end 
    
    if(D(stage)<= S(stage)+U(stage))
        
     Profit(stage) =   -(m*S(stage)+c*U(stage))+r*D(stage);
        
    end 
end

disp('Profit')
disp(sum(Profit))



