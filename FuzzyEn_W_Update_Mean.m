function W_Grad = FuzzyEn_W_Update_Mean(Y,W,H,m,p,r) %#codegen
%% Update of W matrix using a fuzzy entropy distance measure
%
% Inputs:
%           Y:  Data matrix [X by N]
%           W:  Decomposition matrix [X by K]
%           H:  Decomposition matrix [K by N]
%           m:  Length of sequences to be compared
%           p:  Gradient of the exponential boundary
%           r:  Vector of widths of exponential boundary for each row of Y
%
% Output:
%           W_Grad: Matrix of gradients of each element of W [X by K]
%
%   codegen FuzzyEn_W_Update_Mean -args {coder.typeof(double(0),[100 100000],1),coder.typeof(double(0),[100 100],1),coder.typeof(double(0),[100 100000],1),0,0,coder.typeof(double(0),[100 1],1)} -report -o FuzzyEn_W_Update_Mean_mex

% Calculate the dimensions
[X,N] = size(Y);
[~,K] = size(W);

% Calculate the current value of the approximation of Y
WH = W*H;

% Initialize variables
gradPhi_W   = zeros(X,K);                               % Gradient of Phi(m) for each element of W
gradPhi1_W  = zeros(X,K);                               % Gradient of Phi(m+1) for each element of W
W_Grad      = zeros(X,K);                               % Overall gradient of each element of W
c           = 0;

% Update W
for x = 1:X                                             % Iterate through each row of Y & W
    Phi         = 0;                                    % Initialize variables for similarity of current row
    Phi1        = 0;
    dPhi_x      = zeros(K,1);                           % Initialize variables for gradient of similarity of elements of current row
    dPhi1_x     = zeros(K,1);
    for k=1:K                                           % Iterate through elements of current row
        for i=1:N-m                                     % Iterate through all sequences in current row of Y
            % Select m and m+1 elements of current row of Y starting from i
            y_m     = Y(x,i:i+m-1);
            y_m     = y_m - mean(y_m);
            y_m1    = Y(x,i:i+m);
            y_m1    = y_m1 - mean(y_m1);
            for j=1:N-m                                 % Iterate through all sequences in current row of WH
                % Select m and m+1 elements of current row of WH starting from j
                wh_m    = WH(x,j:j+m-1);
                wh_m    = wh_m - mean(wh_m);
                wh_m1   = WH(x,j:j+m);
                wh_m1   = wh_m1 - mean(wh_m1);
                
                % Calculate the current distances and similarity of vectors
                d_m     = norm(y_m-wh_m,Inf);
                d_m1    = norm(y_m1-wh_m1,Inf);
                D_m     = exp(-d_m^p/r(x));
                D_m1    = exp(-d_m1^p/r(x));
                
                if k==1
                    % On the first iteration calculate the average degree of similarity of all the vectors in the row
                    Phi     = Phi + D_m;
                    Phi1    = Phi1 + D_m1;
                end
              
                % Calculate the values of the gradient of the similarity
                c(1)        = find(abs(y_m-wh_m) == d_m,1,'first');
                dPhi_x(k)   = dPhi_x(k) + D_m*(p/r(x))*d_m^(p-1)*(H(k,j+c-1)-mean(H(k,j:j+m-1)))*sign(y_m(c)-wh_m(c));
                c(1)        = find(abs(y_m1-wh_m1) == d_m1,1,'first');
                dPhi1_x(k)  = dPhi1_x(k) + D_m1*(p/r(x))*d_m1^(p-1)*(H(k,j+c-1)-mean(H(k,j:j+m)))*sign(y_m1(c)-wh_m1(c));
            end
        end
    end
    if Phi~=0
        % Calculate the gradient of the row of W for m and m+1
        gradPhi_W(x,:)  = dPhi_x/Phi;
    end
    if Phi1~=0
        gradPhi1_W(x,:) = dPhi1_x/Phi1;
    end
    
    % Calculate the overal gradient of the element of W
    W_Grad(x,:) = gradPhi1_W(x,:) - gradPhi_W(x,:);
end