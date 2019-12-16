function [W_final,H_final,tol,err,err_final] = FuzzyEn_Mean_HCol_NMF(Y,W,H,m,p,r,eta_w,eta_h)
%% Calcluate the nonnegative matrix factorization of Y using a fuzzy entropy distance measure
%
% Inputs:
%           Y           Data matrix [X by N]
%           W           Initial estimate of decomposition matrix [X by K]
%           H           Initial estimate of decomposition matrix [K by N]
%           m           Length of sequences to be compared
%           p           Gradient of the exponential boundary
%           r           Width of the exponential boundary
%           eta_w       Learning rate of W
%           eta_h       Learning rate of H
% 
% Outputs
%           W           Estimated demixing matrix
%           H           Estimated demixing matrix
%           tol         Tolerance between Y - W*H for current and previous iterations
%           err         Error between Y & estimate of W*H
%           err_final   The smallest error between Y & W*H

% Set algorithm variables
% eps         = 1e-4;         % Value at which we judge the tolerance between successive iterations to have no significant update
itr         = 100;          % Maximum number of iterations to try
tol         = zeros(1,itr); % Initialize the tolerance vector
err         = zeros(1,itr); % Initialize the error vector
err_final   = norm(Y-W*H);  % Set the current error value
W_final     = W;
H_final     = H;

CFuzzyEn_YWH    = zeros(itr,size(Y,1));
% r_row       = r*std(Y,0,2); % Adjust r by the standard deviation of each row of Y
r_row       = r*ones(size(Y,1),1);
% r_col       = r*std(Y,0,1); % Adjust r by the standard deviation of each column of Y
r_col       = r*ones(1,size(Y,2),1);

% Iterate for the maximum number of iterations or until the tolerance is no longer changing
for i=1:itr
    
    % Calculate current gradient of W
    W_Grad = FuzzyEn_W_Update_Mean_mex(Y,W,H,m,p,r_row);
    
    % Update W
    W_1     = W;
    W       = W + eta_w*W_Grad;
    W(W<0)  = 1e-16;
    
    % Calculate current gradient of H
    H_Grad = FuzzyEn_H_Update_Col_Mean_mex(Y,W,H,m,p,r_col);
    
    % Update H
    H_1     = H;
    H       = H + eta_h*H_Grad;
    H(H<0)  = 1e-16;
    
    % Calculate current CFuzzyEn
    CFuzzyEn_YWH(i,:) = CFuzzyEn_row_mex(Y,W,H,m,p,r);
    
    % Check if difference between current estimate and Y and the previous estimate and Y is significant
    tol(i) = (norm(Y-W*H,'fro')- norm(Y-W_1*H_1,'fro'))/norm(Y-W*H,'fro');
%     err(i) = norm(Y-W*H);
    err(i) = norm(CFuzzyEn_YWH(i,:));
    fprintf('Iteration: %d \t Tolerance: %d \t Error: %d\n',i,tol(i),err(i));
%     if tol(i)<eps
%         break;
%     end  

    % If the current error is less than the current smallest error update the final error & W & H values
    if err(i)<err_final
        err_final   = err(i);
        W_final     = W;
        H_final     = H;
    end
end
