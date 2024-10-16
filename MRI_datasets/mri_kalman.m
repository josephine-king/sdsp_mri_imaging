functions = common_functions;

[bad_channel1, bad_channel2, bad_channel3] = functions.get_data("1", 0);
corrupted_pixels = functions.get_corrupted_pixels(bad_channel1, 0.9, 105);
bad_channel = functions.fuse_channels_wiener(bad_channel1, bad_channel2, bad_channel3);

kalman_n_rows = 5;
kalman_n_cols = 5;
kalman_dims = [kalman_n_rows,kalman_n_cols];

variance = functions.calculate_variance(bad_channel1,1/16);
Q1 = variance*eye(kalman_n_rows*kalman_n_cols);
estimated_channel1 = kalman_filter_2d(bad_channel1, corrupted_pixels, Q1, kalman_dims);

variance = functions.calculate_variance(bad_channel2,1/16);
Q2 = variance*eye(kalman_n_rows*kalman_n_cols);
estimated_channel2 = kalman_filter_2d(bad_channel2, corrupted_pixels, Q2, kalman_dims);

variance = functions.calculate_variance(bad_channel3,1/16);
Q3 = variance*eye(kalman_n_rows*kalman_n_cols);
estimated_channel3 = kalman_filter_2d(bad_channel3, corrupted_pixels, Q3, kalman_dims);

fused_estimated_channels = functions.fuse_channels_wiener(estimated_channel1, estimated_channel2, estimated_channel3);

bad_img = functions.get_image(bad_channel);
bad_adj_img = functions.adjust_image(bad_img, 1);

corr_img = functions.get_image(fused_estimated_channels);
corr_adj_img = functions.adjust_image(corr_img, 1);


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
imagesc(100*log(abs(bad_channel)));
subplot(1,2,2)
imagesc(100*log(abs(fused_estimated_channels)));
title("Good image, fusion after IFFT")


%%

function estimated_channel = kalman_filter_2d(channel, corrupted_pixels, Q, dims)
    n_rows = dims(1);
    n_cols = dims(2);
    half_n_rows = floor(n_rows / 2);
    half_n_cols = floor(n_cols / 2);
    num_rows = size(channel, 1);
    num_cols = size(channel, 2);

    % Initialize the estimated channel
    estimated_channel = zeros(size(channel));

    % Initialize the state vector using the first window
    x = reshape(channel(1:half_n_rows*2+1, 1:half_n_cols*2+1), [], 1);

    % Initialize error covariance matrix as identity
    P = 1000*eye(n_rows * n_cols);

    % State transition matrix (identity for simplicity)
    F = eye(n_rows * n_cols);

    % Observation matrix
    C = zeros(1, n_rows * n_cols);
    C(half_n_rows * n_cols + half_n_cols + 1) = 1; % Center pixel observation

    for k_row = 1 + half_n_rows:num_rows - half_n_rows
        for k_col = 1 + half_n_cols:num_cols - half_n_cols
            
            if corrupted_pixels(k_row, k_col) == 1
                estimated_channel(k_row, k_col) = C * x;
            else
            
            % Extract the 2D window
            window = channel((k_row - half_n_rows):(k_row + half_n_rows), ...
                             (k_col - half_n_cols):(k_col + half_n_cols));

            % Prediction step
            x_predict = F * x;  % Predict next state
            P_predict = F * P * F' + Q;  % Predict error covariance

            % Update step (when measurement is available)
            z_k = window(half_n_rows + 1, half_n_cols + 1);  % Measured center pixel
            y = z_k - C * x_predict;  % Innovation

            % Kalman gain
            K = P_predict * C' / (C * P_predict * C' + 1e-6);  % Add small value for numerical stability

            % Update state estimate
            x = x_predict + K * y;

            % Update covariance
            P = (eye(size(P_predict)) - K * C) * P_predict;

            % Store results for corrupted pixels

                estimated_channel(k_row, k_col) = channel(k_row, k_col);
            end
        end
    end
    
    estimated_channel(1:half_n_rows, :) = channel(1:half_n_rows, :);
    estimated_channel(end-half_n_rows+1:end, :) = channel(end-half_n_rows+1:end, :);
    estimated_channel(:, 1:half_n_cols) = channel(:, 1:half_n_cols);
    estimated_channel(:, end-half_n_cols+1:end) = channel(:, end-half_n_cols+1:end);
end




