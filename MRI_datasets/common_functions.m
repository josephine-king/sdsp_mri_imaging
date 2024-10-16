classdef common_functions
methods(Static)

% Loads the data from slice slice_num
% If good_n_bad is 1, it will load the good data. Otherwise it'll load the
% bad data. The three channels are stored in [channel1, channel2, channel3]
function [channel1, channel2, channel3] = get_data(slice_num, good_n_bad)

    if (good_n_bad == 1)
        good_str = 'Good';
    else 
        good_str = 'Bad';
    end
    load(strcat('Slice',slice_num,'/',good_str,'Data/slice',slice_num,'_channel1.mat'));
    load(strcat('Slice',slice_num,'/',good_str,'Data/slice',slice_num,'_channel2.mat'));
    load(strcat('Slice',slice_num,'/',good_str,'Data/slice',slice_num,'_channel3.mat'));
    
    if (good_n_bad == 1)
        good_str = 'good';
    else 
        good_str = 'bad';
    end
    channel1 = eval(strcat('slice',slice_num,'_channel1_',good_str,'Data'));
    channel2 = eval(strcat('slice',slice_num,'_channel2_',good_str,'Data'));
    channel3 = eval(strcat('slice',slice_num,'_channel3_',good_str,'Data'));

end

% Identifies the corrupted pixels. Returns a matrix with the same size as
% channel. Corrupted pixels are 1, non corrupted are 0
function corrupted_pixels1  = get_corrupted_pixels(channel, threshold1, threshold2)
    num_rows = size(channel,1);
    num_cols = size(channel,2);
    corrupted_pixels = zeros(num_rows,num_cols);
    row_ns = 3;
    col_ns = 1;
    for col = 1+col_ns:num_cols-col_ns
        for row = 1+row_ns:num_rows-row_ns
            m = channel(row-row_ns:row+row_ns, col-col_ns:col+col_ns);
            this_col = m(:,col_ns+1);
            m(:,col_ns+1) = [];
            Pm = mean(abs(m).^2);
            P_this_col = mean(abs(this_col).^2);

            r = abs(Pm-P_this_col)/Pm;
 
            if (r > threshold1)
                corrupted_pixels(row,col) = r;
            end
        end
        if (norm(corrupted_pixels(:,col),1) > threshold2)
            lower = 1;%round(size(corrupted_pixels,1)/8);
            upper = size(corrupted_pixels,1);%round(7*size(corrupted_pixels,1)/8);
            corrupted_pixels(:,col) = zeros(size(corrupted_pixels,1),1);
            corrupted_pixels(lower:upper,col) = ones(upper-lower+1,1);
        else
            corrupted_pixels(:,col) = zeros(size(corrupted_pixels,1),1);
        end
    end

    corrupted_pixels1 = zeros(num_rows,num_cols);
    corrupted_lines = [6,22,38,54,70,86,102,118];
    for idx = 1:length(corrupted_lines)
        corrupted_pixels1(:,corrupted_lines(idx)) = ones(512,1);
    end
end

% ---------------------------------------------------------------------
% Channel fusion functions 
% ---------------------------------------------------------------------
% Fuses channels using a simple method based on the noise and signal
% variances
function fused_channels = fuse_channels_simple(channel1, channel2, channel3)
    [var_v1, var_c1] = common_functions.calculate_variance(channel1, 1/16);
    [var_v2, var_c2] = common_functions.calculate_variance(channel2, 1/16);
    [var_v3, var_c3] = common_functions.calculate_variance(channel3, 1/16);
    Rx = [var_c1, 0, 0; 0, var_c2, 0; 0, 0, var_c3];
    rdx = [var_c1-var_v1; var_c2-var_v2; var_c3-var_v3];
    w = inv(Rx)*rdx;
    fused_channels = (w(1)*channel1+w(2)*channel2+w(3)*channel3);
end

function fused_channels = fuse_channels_wiener(channel1, channel2, channel3)
    [Rx, r_dx] = common_functions.get_Rx_rdx_for_fusion(channel1, channel2, channel3);
    w = inv(Rx)*r_dx;
    fused_channels = (w(1)*channel1+w(2)*channel2+w(3)*channel3);
end

function [Rx, r_dx] = get_Rx_rdx_for_fusion(channel1, channel2, channel3)
    
    num_rows = size(channel1, 1);
    num_cols = size(channel1,2);

    % Initialize matrices for both cross-correlation and autocorrelation
    row_rdx = [];
    row_Rs = [];

    for row = 1:num_rows
        col_rdx = [];
        col_Rs = [];
        for col = 1:num_cols

            pixel1 = channel1(row,col);
            pixel2 = channel2(row,col);
            pixel3 = channel3(row,col);
            v = [pixel1; pixel2; pixel3];
            d = (pixel1 + pixel2 + pixel3)/3;

            % Autocorrelation computation
            R = v * ctranspose(v);
            col_Rs = [col_Rs, reshape(R, [], 1)];

            % Cross-correlation computation
            curr_rdx = d * conj(v);
            col_rdx = [col_rdx, curr_rdx];

        end

        avg_col_rdx = mean(col_rdx, 2);
        row_rdx = [row_rdx, avg_col_rdx];

        avg_col_Rs = mean(col_Rs, 2);
        row_Rs = [row_Rs, avg_col_Rs];

    end

    r_dx = mean(row_rdx, 2);
    Rx = mean(row_Rs, 2);
    Rx = reshape(Rx, 3, 3);

end

% ---------------------------------------------------------------------
% Statistical estimation functions
% ---------------------------------------------------------------------
function [Rx, r_dx] = get_Rx_rdx(channel, window_dims, corrupted_pixels, center_n_edge, center_dims, omit_corrupted_col)
    
    window_rows = window_dims(1);
    window_cols = window_dims(2);

    center_start_rows = max(round((size(channel, 1) - center_dims(1)) / 2), 1);
    center_start_cols = max(round((size(channel, 2) - center_dims(2)) / 2));
    center_end_rows = center_start_rows + center_dims(1);
    center_end_cols = center_start_cols + center_dims(2);

    % Initialize matrices for both cross-correlation and autocorrelation
    row_rdx = [];
    row_Rs = [];

    window_start_rows = 1;
    while (window_start_rows + window_rows - 1 <= size(channel, 1))
        col_rdx = [];
        col_Rs = [];

        window_row_range = window_start_rows:window_start_rows + window_rows - 1;
        if (window_rows == 1)
            window_row_in_center = (window_start_rows >= center_start_rows && window_start_rows <= center_end_rows);
        else
            window_row_in_center = min(window_start_rows + window_rows - 1, center_end_rows) > max(window_start_rows, center_start_rows);
        end
        window_start_cols = 1;

        while (window_start_cols + window_cols - 1 <= size(channel, 2))
            window_end_rows = window_start_rows + window_rows - 1;
            window_end_cols = window_start_cols + window_cols - 1;
            window_col_range = window_start_cols:window_start_cols + window_cols - 1;
            if (window_cols == 1) 
                window_col_in_center = (window_start_cols >= center_start_cols && window_start_cols <= center_end_cols);
            else 
                window_col_in_center = min(window_start_cols + window_cols - 1, center_end_cols) > max(window_start_cols, center_start_cols);
            end
            window_in_center = window_row_in_center && window_col_in_center;

            if (center_n_edge && ~window_in_center) 
                window_start_cols = window_start_cols + 1; 
                continue; 
            end
            if (~center_n_edge && window_in_center) 
                window_start_cols = window_start_cols + 1; 
                continue; 
            end
         
            m = channel(window_start_rows:window_end_rows, window_start_cols:window_end_cols);
            d = m(round((window_rows + 1) / 2), round((window_cols + 1) / 2));
            if (omit_corrupted_col)
                m(:, round((window_cols + 1) / 2)) = []; 
            end
            v = reshape(m, [], 1);

            % Check if there are any corrupted pixels before computing Rx.
            % The center row doesn't matter as it's not part of the calculation
            cps = corrupted_pixels(window_row_range, window_col_range);
            cps(:,round(window_cols+1)/2) = [];
            if (any(cps, 'all'))
                window_start_cols = window_start_cols+1;
                continue; 
            end
            % Autocorrelation computation
            R = v * ctranspose(v);
            col_Rs = [col_Rs, reshape(R, [], 1)];

            % Check if there are any corrupted pixels before computing Rx
            if (any(corrupted_pixels(window_row_range, window_col_range), 'all'))
                window_start_cols = window_start_cols + 1; 
                continue; 
            end
            % Cross-correlation computation
            curr_rdx = d * conj(v);
            col_rdx = [col_rdx, curr_rdx];

            window_start_cols = window_start_cols + 1;
        end

        avg_col_rdx = mean(col_rdx, 2);
        row_rdx = [row_rdx, avg_col_rdx];

        avg_col_Rs = mean(col_Rs, 2);
        row_Rs = [row_Rs, avg_col_Rs];

        window_start_rows = window_start_rows + 1;
    end

    r_dx = mean(row_rdx, 2);
    Rx = mean(row_Rs, 2);
    R_rows = window_rows*window_cols-omit_corrupted_col*window_rows;
    R_cols = window_rows*window_cols-omit_corrupted_col*window_rows;
    Rx = reshape(Rx, R_rows, R_cols);
end

function [var_v, var_c] = calculate_variance(channel, noise_img_frac)
    % get variance of the whole channel
    var_c = var(reshape(channel, 1, []));

    % corner of the image is assumed to have the most noise
    rows_dim = size(channel,1);
    cols_dim = size(channel,2);
    noise_area = channel(1:round(noise_img_frac*rows_dim),1:round(noise_img_frac*cols_dim));
    var_v = var(reshape(noise_area, 1, []));
end

% ---------------------------------------------------------------------
% Image processing functions 
% ---------------------------------------------------------------------
% Obtain the image from 3 channels of k-space data using IFFT

function [img, MSE] = get_image_and_mse(channels, true_img)
    img = common_functions.get_image(channels);
    img = common_functions.adjust_image(img,1);
    MSE = norm(double(true_img) - double(img),2);
end

function [img, MSE] = get_image_and_mse_nonfused(channel1, channel2, channel3, true_img)
    img = common_functions.get_image_non_fused(channel1, channel2, channel3);
    img = common_functions.adjust_image(img,0);
    MSE = norm(double(true_img) - double(img),2);
end

% Obtain the image from the k-space data using IFFT
function img = get_image(chan)
    img = ifftshift(ifft2(chan),1);
end

% Obtain the image from 3 channels of k-space data using IFFT
function img = get_image_non_fused(chan1, chan2, chan3)
    img(:,:,1) = ifftshift(ifft2(chan1),1);
    img(:,:,2) = ifftshift(ifft2(chan2),1);
    img(:,:,3) = ifftshift(ifft2(chan3),1);
end

% Do some post processing on the image
function adj_img = adjust_image(img, fused)
    % clear compensation, preparation, based on fourier transformed blinked 
    % k-space data (Data_raw)
    clear_comp = linspace(10,0.1,size(img,2)).^2; 
    clear_matrix = repmat(clear_comp,[size(img,1) 1]);

    % combine 3 channels sum of squares and add clear compensation
    if (fused == 0)
        adj_img  = sqrt( abs(squeeze(img(:,:,1))).^2 + ...
                     abs(squeeze(img(:,:,2))).^2 + ...
                     abs(squeeze(img(:,:,3))).^2).* clear_matrix; 
    else
        adj_img = abs(squeeze(img)).*clear_matrix;
    end
    
    % crop images because we are only interested in eye 
    dims = size(adj_img(:,:,1));
    y_dim = dims(1);
    x_dim = dims(2);
    lower_y_bound = (y_dim - x_dim)/2;
    upper_y_bound = lower_y_bound + x_dim;
    crop_y = [lower_y_bound : upper_y_bound]; % crop coordinates
    adj_img = adj_img(crop_y, :);
    adj_img_dims = size(adj_img);

    %image
    adj_img = reshape(squeeze(adj_img(:,:)),[adj_img_dims(1) adj_img_dims(2)]); 

    % For better visualization and contrast of the eye images, histogram based
    % compensation will be done 
    std_within = 0.995; 
    % set maximum intensity to contain 99.5 % of intensity values per image
    [aa, val] = hist(adj_img(:),linspace(0,max(adj_img(:)),1000));
        thresh = val(find(cumsum(aa)/sum(aa) > std_within,1,'first'));
    
    % set threshold value to 65536
    adj_img = uint16(adj_img * 65536 / thresh); 
end


end
end