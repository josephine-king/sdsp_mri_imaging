function filtered_channels = mri_kalman(channel1, channel2, channel3, cp1, cp2, cp3, fuse_first, fuse_type, kalman_dims)
    functions = common_functions;
    if (fuse_first == 1)
        if (fuse_type == "simple")
            channels = functions.fuse_channels_simple(channel1,channel2,channel3);
        else
            channels = functions.fuse_channels_wiener(channel1,channel2,channel3);
        end
        variance = functions.calculate_variance(channels,1/16);
        Q = variance*eye(kalman_dims(1)*kalman_dims(2));
        filtered_channels = kalman_filter_2d(channels, cp1, Q, kalman_dims);
    else
        variance = functions.calculate_variance(channel1,1/16);
        Q = variance*eye(kalman_dims(1)*kalman_dims(2));
        filtered_channel1 = kalman_filter_2d(channel1, cp1, Q, kalman_dims);

        variance = functions.calculate_variance(channel2,1/16);
        Q = variance*eye(kalman_dims(1)*kalman_dims(2));
        filtered_channel2 = kalman_filter_2d(channel2, cp2, Q, kalman_dims);

        variance = functions.calculate_variance(channel3,1/16);
        Q = variance*eye(kalman_dims(1)*kalman_dims(2));
        filtered_channel3 = kalman_filter_2d(channel3, cp3, Q, kalman_dims);

        if (fuse_type == "simple")
            filtered_channels = functions.fuse_channels_simple(filtered_channel1,filtered_channel2,filtered_channel3);
        else
            filtered_channels = functions.fuse_channels_wiener(filtered_channel1,filtered_channel2,filtered_channel3);
        end
    end
end

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

    % Initialize error covariance matrix as identity, multiplied by 1000 to
    % show high initial uncertainty
    P = 5000*eye(n_rows * n_cols);

    % State transition matrix (identity for simplicity)
    F = eye(n_rows*n_cols);

    % Observation matrix
    C = eye(n_rows * n_cols);

    for k_row = 1 + half_n_rows:num_rows - half_n_rows
        for k_col = 1 + half_n_cols:num_cols - half_n_cols
            % Store results for corrupted pixels
            if corrupted_pixels(k_row, k_col) == 1
                estimated_channel(k_row, k_col) = x(half_n_rows*n_cols + half_n_cols + 1);
            else
                % Extract the 2D window measurement
                z_k = channel((k_row - half_n_rows):(k_row + half_n_rows), (k_col - half_n_cols):(k_col + half_n_cols));
                z_k = reshape(z_k,[],1);
                
                % Prediction step
                x_predict = F * x;  % Predict next state
                P_predict = F * P * F' + Q;  % Predict error covariance

                % Update step (when measurement is available)
                y = z_k - C * x_predict;  % Innovation

                % Kalman gain
                K = P_predict * C' / (C * P_predict * C' + 1e-6);  % Add small value for numerical stability

                % Update state estimate
                x = x_predict + K * y;

                % Update covariance
                P = (eye(size(P_predict)) - K * C) * P_predict;

                estimated_channel(k_row, k_col) = channel(k_row, k_col);
            end

        end
    end
    
    estimated_channel(1:half_n_rows, :) = channel(1:half_n_rows, :);
    estimated_channel(end-half_n_rows+1:end, :) = channel(end-half_n_rows+1:end, :);
    estimated_channel(:, 1:half_n_cols) = channel(:, 1:half_n_cols);
    estimated_channel(:, end-half_n_cols+1:end) = channel(:, end-half_n_cols+1:end);
end




