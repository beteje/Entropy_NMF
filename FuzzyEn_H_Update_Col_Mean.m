function H_Grad = FuzzyEn_H_Update_Col_Mean(Y,W,H,m,p,r) %#codegen
%% Update of W matrix using a fuzzy entropy distance measure
%
% Inputs:
%           Y:  Data matrix [X by N]
%           W:  Decomposition matrix [X by K]
%           H:  Decomposition matrix [K by N]
%           m:  Length of sequences to be compared
%           p:  Gradient of the exponential boundary
%           r:  Vector of widths of exponential boundary for each column of Y
%
% Output:
%           H_Grad: Matrix of gradients of each element of W [X by K]
%
%   codegen FuzzyEn_H_Update_Col_Mean -args {coder.typeof(double(0),[100 100000],1),coder.typeof(double(0),[100 100],1),coder.typeof(double(0),[100 100000],1),0,0,coder.typeof(double(0),[1 100000],1)} -report -o FuzzyEn_H_Update_Col_Mean_mex

% Calculate the dimensions
[X,N] = size(Y);
[~,K] = size(W);

% Calculate the current value of the approximation of Y
WH = W*H;

% Initialize variables
gradPhi_H   = zeros(K,N);                               % Gradient of Phi(m) for each element of W
gradPhi1_H  = zeros(K,N);                               % Gradient of Phi(m+1) for each element of W
H_Grad      = zeros(K,N);                               % Overall gradient of each element of W
c           = 0;

% Update W
for n = 1:N                                             % Iterate through each row of Y & W
    Phi         = 0;                                    % Initialize variables for similarity of current column
    Phi1        = 0;
    dPhi_n      = zeros(K,1);                           % Initialize variables for gradient of similarity of elements of current column
    dPhi1_n     = zeros(K,1);
    for k=1:K                                           % Iterate through elements of current column
        for i=1:X-m                                     % Iterate through all sequences in current row of Y
            % Select m and m+1 elements of current column of Y starting from i
            y_m     = Y(i:i+m-1,n);
            y_m     = y_m - mean(y_m);
            y_m1    = Y(i:i+m,n);
            y_m1    = y_m1 - mean(y_m1);
            for j=1:X-m                                 % Iterate through all sequences in current column of WH
                % Select m and m+1 elements of current row of WH starting from j
                wh_m    = WH(j:j+m-1,n);
                wh_m    = wh_m - mean(wh_m);
                wh_m1   = WH(j:j+m,n);
                wh_m1   = wh_m1 - mean(wh_m1);
                
                % Calculate the current distances and similarity of vectors
                d_m     = norm(y_m-wh_m,Inf);
                d_m1    = norm(y_m1-wh_m1,Inf);
                D_m     = exp(-d_m^p/r(n));
                D_m1    = exp(-d_m1^p/r(n));
                
                if k==1
                    % On the first iteration calculate the average degree of similarity of all the vectors in the column
                    Phi     = Phi + D_m;
                    Phi1    = Phi1 + D_m1;
                end
              
                % Calculate the values of the gradient of the similarity
                c(1)        = find(abs(y_m-wh_m) == d_m,1,'first');
                dPhi_n(k)   = dPhi_n(k) + D_m*(p/r(n))*d_m^(p-1)*sign(y_m(c)-wh_m(c))*(W(j+c-1,k)-mean(W(j:j+m-1,k)));
                c(1)        = find(abs(y_m1-wh_m1) == d_m1,1,'first');
                dPhi1_n(k)  = dPhi1_n(k) + D_m1*(p/r(n))*d_m1^(p-1)*sign(y_m1(c)-wh_m1(c))*(W(j+c-1,k)-mean(W(j:j+m,k)));
            end
        end
    end
    if Phi~=0
        % Calculate the gradient of the column of W for m and m+1
        gradPhi_H(:,n)  = dPhi_n/Phi;
    end
    if Phi1~=0
        gradPhi1_H(:,n) = dPhi1_n/Phi1;
    end
    
    % Calculate the overal gradient of the element of W
    H_Grad(:,n) = gradPhi1_H(:,n) - gradPhi_H(:,n);
end