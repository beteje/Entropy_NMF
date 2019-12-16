function H_Grad = FuzzyEn_H_Update_Mean(Y,W,H,m,p,r) %#codegen
%% Update of H matrix using a fuzzy entropy distance measure
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
%           H_Grad: Matrix of gradients of each element of W [X by K]
%
%   codegen FuzzyEn_H_Update_Mean -args {coder.typeof(double(0),[100 100000],1),coder.typeof(double(0),[100 100],1),coder.typeof(double(0),[100 100000],1),0,0,coder.typeof(double(0),[1 100000],1)} -report -o FuzzyEn_H_Update_Mean_mex

% Calculate the dimensions
[X,N] = size(Y);
[~,K] = size(W);

% calculate the current value of the approximation of Y
WH = W*H;

% Initialize variables
Phi         = zeros(X,1);                               % Phi(m) for each row of Y & WH
Phi1        = zeros(X,1);                               % Phi(m+1) for each row of Y & WH
gradPhi_H   = zeros(K,N);                               % Gradient of Phi(m) for each element of H
gradPhi1_H  = zeros(K,N);                               % Gradient of Phi(m+1) for each element of H
H_Grad      = zeros(K,N);                               % Overall gradient of each element of H
c           = 0;

% Update H
for n=1:N                                               % Iterate through each column of H
    for k=1:K                                           % Iterate through each row of H
        gradPhi_tmp = 0;                                % Initialize temporary variables for current elements gradient
        gradPhi1_tmp = 0;
        for x = 1:X                                     % Iterate through each row of Y & W
            dPhi_x = 0;                                 % Initialize variables for gradient of current row
            dPhi1_x = 0;
            for i=1:N-m                                 % Iterate through all sequences in current row of Y
                % Select m and m+1 elements of current row of Y starting from i
                y_m     = Y(x,i:i+m-1);
                y_m     = y_m - mean(y_m);
                y_m1    = Y(x,i:i+m);
                y_m1    = y_m1 - mean(y_m1);
                
                % Check if first iteration
                if n==1 && k==1
                    % If so iterate through all j in the first row
                    for j=1:N-m
                        % Select m and m+1 elements of current row of WH starting from j
                        wh_m    = WH(x,j:j+m-1);
                        wh_m    = wh_m - mean(wh_m);
                        wh_m1   = WH(x,j:j+m);
                        wh_m1   = wh_m1 - mean(wh_m1);
                        
                        % Calculate the current distances and similarity of vectors
                        d_m  = norm(y_m-wh_m,Inf);
                        d_m1 = norm(y_m1-wh_m1,Inf);
                        D_m = exp(-d_m^p/r(x));
                        D_m1 = exp(-d_m1^p/r(x));
                        
                        % On the first iteration calculate the average degree of similarity of all the vectors
                        Phi(x)  = Phi(x) + D_m;
                        Phi1(x) = Phi1(x) + D_m1;
                        
                        z = n-j+1;                          % Find the position of the current element in the vectors
                        if j>=max(1,n-m+1) && j<=min(n,N-m) % Check whether the current element is in the current vector of length m
                            c(1) = find(abs(y_m-wh_m) == d_m,1,'first');
                            % Check whether the current element contributes to the maximum distance
                            if (c == z)
                                % If so calculate the values of the gradient of the similarity
                                dPhi_x = dPhi_x + D_m*(p/r(x))*d_m^(p-1)*((m-1)/m)*sign(y_m(c)-wh_m(c))*W(x,k);         
                            else
                                dPhi_x = dPhi_x + D_m*(p/r(x))*d_m^(p-1)*(1/m)*sign(y_m(c)-wh_m(c))*W(x,k);
                            end
                            
                            if j>=max(1,n-m) && j<=min(n,N-m) % Check whether the current element is in the current vector of length m+1
                                c(1) = find(abs(y_m1-wh_m1) == d_m1,1);
                                % Check whether the current element contributes to the maximum distance
                                if (c == z)
                                    % If so calculate the values of the gradient of the similarity
                                    dPhi1_x = dPhi1_x + D_m1*(p/r(x))*d_m1^(p-1)*(m/(m+1))*sign(y_m1(c)-wh_m1(c))*W(x,k);
                                else
                                    dPhi1_x = dPhi1_x + D_m1*(p/r(x))*d_m1^(p-1)*(1/(m+1))*sign(y_m1(c)-wh_m1(c))*W(x,k);
                                end
                            end
                        end
                    end
                else
                    % If not the first iteration select only the j (n-m) either side of n
                    for j=max(1,n-m):min(n,N-m)
                        z = n-j+1;                          % Find the position of the current element in the vectors
                        if j>=max(1,n-m+1)                  % Check whether the current element is in the current vector of length m
                            % Select m elements of current row of WH starting from j
                            wh_m    = WH(x,j:j+m-1);
                            wh_m    = wh_m - mean(wh_m);
                            
                            % Calculate the current distance and similarity of vectors
                            d_m  = norm(y_m-wh_m,Inf);
                            D_m = exp(-d_m^p/r(x));
                            
                            c(1) = find(abs(y_m-wh_m) == d_m,1);
                            % Check whether the current element contributes to the maximum distance
                            if (c == z)
                                % Calculate the values of the gradient of the similarity
                                dPhi_x = dPhi_x + D_m*(p/r(x))*d_m^(p-1)*((m-1)/m)*sign(y_m(c)-wh_m(c))*W(x,k);
                            else
                                dPhi_x = dPhi_x + D_m*(p/r(x))*d_m^(p-1)*(1/m)*sign(y_m(c)-wh_m(c))*W(x,k);
                            end
                        end
                        
                        % Select m+1 elements of current row of WH starting from j
                        wh_m1   = WH(x,j:j+m);
                        wh_m1   = wh_m1 - mean(wh_m1);
                        
                        % Calculate the current distances and similarity of vectors
                        d_m1 = norm(y_m1-wh_m1,Inf);
                        D_m1 = exp(-d_m1^p/r(x));
                        
                        c(1) = find(abs(y_m1-wh_m1) == d_m1,1);
                        % Check whether the current element contributes to the maximum distance
                        if (c == z)
                            % Calculate the values of the gradient of the similarity
                            dPhi1_x = dPhi1_x + D_m1*(p/r(x))*d_m1^(p-1)*(m/(m+1))*sign(y_m1(c)-wh_m1(c))*W(x,k);
                        else
                            dPhi1_x = dPhi1_x + D_m1*(p/r(x))*d_m1^(p-1)*(1/(m+1))*sign(y_m1(c)-wh_m1(c))*W(x,k);
                        end
                    end
                end
            end
            if Phi(x)~=0
                % Calculate the gradient for the current element across the rows of WH for m and m+1
                gradPhi_tmp = gradPhi_tmp + dPhi_x/Phi(x);
            end
            if Phi1(x)~=0
                gradPhi1_tmp = gradPhi1_tmp + dPhi1_x/Phi1(x);
            end
        end
        % Store the gradients for the current element for m and m+1
        gradPhi_H(k,n) = gradPhi_tmp;
        gradPhi1_H(k,n) = gradPhi1_tmp;
        % Calculate the overal gradient of the element of
        H_Grad(k,n) = gradPhi1_H(k,n) - gradPhi_H(k,n);
    end
end