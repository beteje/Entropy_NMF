function CFuzzyEn_x = CFuzzyEn_row(Y,W,H,m,p,r) %codegen
%% Calculates the cross fuzzy entropy between the rows of Y and W*H 

% Calculate the dimensions
[X,N] = size(Y);

% Calculate the estimate of Y
WH = W*H;
% Calculate row and column values of r
r_row   = r*std(Y,0,2);

CFuzzyEn_x = zeros(1,X);
for x = 1:X                                             % Iterate through each row of Y & W
    Phi_x         = 0;                                  % Initialize variables for similarity of current row
    Phi1_x        = 0;
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
                D_m     = exp(-d_m^p/r_row(x));
                D_m1    = exp(-d_m1^p/r_row(x));
                
                % Calculate the average degree of similarity of all the vectors in the row
                Phi_x     = Phi_x + D_m;
                Phi1_x    = Phi1_x + D_m1;
            end
        end  
    CFuzzyEn_x(x) = log(Phi_x) - log(Phi1_x);
end