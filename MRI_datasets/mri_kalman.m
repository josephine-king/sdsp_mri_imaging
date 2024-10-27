% Filters and fuses three MRI channels using a kalman filter. 
% channel1-3 are the three channels, and cp1-3 are their corrupted pixel
% matrices
% fuse_first - if one, fuse before filtering. Otherwise, fuse after
% filtering
% fuse type - choose between noise, snr, average, or wiener fusion methods
function filtered_channels = mri_kalman(channel1, channel2, channel3, cp1, cp2, cp3, fuse_first, fuse_type)
    functions = common_functions;
    if (fuse_first == 1)
        if (fuse_type == "noise")
            channels = functions.fuse_channels_noise(channel1,channel2,channel3);
        elseif (fuse_type == "snr")
            channels = functions.fuse_channels_snr(channel1,channel2,channel3);
        elseif (fuse_type == "average")
            channels = functions.fuse_channels_average(channel1,channel2,channel3);
        else
            channels = functions.fuse_channels_wiener(channel1,channel2,channel3);
        end
        variance = functions.calculate_variance(channels,1/16);
        filtered_channels = kalman_filter_2d(channels, cp1, variance);
    else
        variance = functions.calculate_variance(channel1,1/16);
        filtered_channel1 = kalman_filter_2d(channel1, cp1, variance);

        variance = functions.calculate_variance(channel2,1/16);
        filtered_channel2 = kalman_filter_2d(channel2, cp2, variance);

        variance = functions.calculate_variance(channel3,1/16);
        filtered_channel3 = kalman_filter_2d(channel3, cp3, variance);

        if (fuse_type == "noise")
            filtered_channels = functions.fuse_channels_noise(filtered_channel1,filtered_channel2,filtered_channel3);
        elseif (fuse_type == "snr")
            filtered_channels = functions.fuse_channels_snr(filtered_channel1,filtered_channel2,filtered_channel3);
        elseif (fuse_type == "average")
            filtered_channels = functions.fuse_channels_average(filtered_channel1,filtered_channel2,filtered_channel3);
        else
            filtered_channels = functions.fuse_channels_wiener(filtered_channel1,filtered_channel2,filtered_channel3);
        end
    end
end

% Applies a Kalman filter by sliding a 2D window over the K-space
% Arguments are the k-space channel, the corrupted pixel matrix, process
% noise covariance matrix Q, and the dimensions of the window
% Returns the filtered channel
function estimated_channel = kalman_filter_2d(channel, corrupted_pixels, var_v)
    n_rows = 1;
    n_cols = 2;
    num_rows = size(channel, 1);
    num_cols = size(channel, 2);

    % Initialize the estimated channel
    estimated_channel = zeros(size(channel));
    % Initialize the state vector using the first window
    x = reshape(channel(1, 1:2), [], 1);

    % Get Q matrices based on the covariance
    Qw = eye(n_rows*n_cols)*var_v;
    Qv = var_v; % one dimensional measurement
    % Initialize error covariance matrix as Q
    P = Qw;

    % State transition matrix (identity for simplicity)
    A = eye(n_rows*n_cols);
    % Observation matrix
    C = [0,1];

    for k_row = 1:num_rows
        for k_col = 2:num_cols
            % Store results for corrupted pixels
            if corrupted_pixels(k_row, k_col) == 1
                x_predict = A * x;  % Predict next state
                P_predict = A * P * A' + Qw;  % Predict error covariance
                x = x_predict;
                P = P_predict;
                estimated_channel(k_row, k_col) = C*x;
            else
                % Extract the 2D window measurement
                z_k = channel(k_row,k_col);
                
                % Prediction step
                x_predict = A * x;  % Predict next state
                P_predict = A * P * A' + Qw;  % Predict error covariance

                % Update step (when measurement is available)
                e = z_k - C * x_predict;  % Innovation
                % Kalman gain
                K = P_predict * C' / (Qv + C * P_predict * C');  % Add small value for numerical stability
                % Update state estimate
                x = x_predict + K * e;
                % Update covariance
                P = (eye(size(P_predict)) - K * C) * P_predict;
                
                % For non corrupted pixels, just use the channel data
                estimated_channel(k_row, k_col) = channel(k_row, k_col);
            end

        end
    end
    
    % Fill in missing pixels around the edges
    estimated_channel(:, 1:2) = channel(:, 1:2);
end




