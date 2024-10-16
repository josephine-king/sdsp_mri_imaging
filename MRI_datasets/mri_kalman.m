functions = common_functions;

[bad_channel1, bad_channel2, bad_channel3] = functions.get_data("4", 0);
corrupted_pixels = functions.get_corrupted_pixels(bad_channel1, 0.9, 105);

Q = eye(3,3)*10;
estimated_channel1 = kalman_filter(bad_channel1, corrupted_pixels, Q);
estimated_channel2 = kalman_filter(bad_channel2, corrupted_pixels, Q);
estimated_channel3 = kalman_filter(bad_channel3, corrupted_pixels, Q);

bad_img = functions.get_image_non_fused(bad_channel1, bad_channel2, bad_channel3);
bad_adj_img = functions.adjust_image(bad_img, 0);

corr_img = functions.get_image_non_fused(estimated_channel1, estimated_channel2, estimated_channel3);
corr_adj_img = functions.adjust_image(corr_img, 0);

figure(3)
axis image, 
colormap gray;
axis off
subplot(1,2,1)
imagesc(bad_adj_img);
subplot(1,2,2)
imagesc(corr_adj_img);
title("Good image, fusion after IFFT")

figure(2)
subplot(1,2,1)
imagesc(100*log(abs(bad_channel1)));
subplot(1,2,2)
imagesc(100*log(abs(estimated_channel1)));
title("Good image, fusion after IFFT")


%%
function estimated_y = kalman_filter(channel, corrupted_pixels, Q)
    % Initialize state vector
    % State vector: [x_left; x_corrupted; x_right]
    n = 3; 
    num_rows = size(channel,1);
    num_cols = size(channel,2);
    channel = reshape(channel, 1, []);
    corrupted_pixels = reshape(corrupted_pixels, 1, []);
    num_iterations = size(channel,2);
    x = channel(1,1:n)'; % Initial guess for x_corrupted can be zero or any initial value
    
    % Initialize error covariance matrix
    P = eye(3); % 3x3 identity matrix for initial covariance

    % State transition matrix
    F = [1, 0, 0;  % x_left remains the same
         0, 1, 0;  % x_corrupted
         0, 0, 1];  % x_right remains the same

    % Observation matrix (observing only the corrupted pixel)
    C = [0, 1, 0];  

    % Preallocate arrays for results
    estimated_y = zeros(1, num_iterations);

    for k = 2:size(channel,2)-1
        % Prediction step
        x_predict = F * x;  % Predict next state
        P_predict = F * P * F' + Q;  % Predict error covariance

        % Update step (when measurement is available)
        z_k = channel(k);  % Your measured value of the corrupted pixel
        y = z_k - C * x_predict;  % Innovation

        % Kalman gain
        K = P_predict * C' / (C * P_predict * C');

        % Update state estimate
        x = x_predict + K * y;

        % Update covariance
        P = (eye(size(P_predict)) - K * C) * P_predict;

        % Store results
        if (corrupted_pixels(k)==1)
            estimated_y(1, k) = y;
        else 
            estimated_y(1, k) = channel(1, k);
        end
    end
    estimated_y(1,1) = channel(1,1);
    estimated_y(1,size(channel,2)) = channel(1,size(channel,2));
    estimated_y = reshape(estimated_y, num_rows, num_cols);
end


