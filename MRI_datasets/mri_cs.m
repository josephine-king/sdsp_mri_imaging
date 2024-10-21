%% Script to translate the K-spcae images into spatial eye images
clc; clear all; close all;

%% Produce 4 images using different methods: 
% Simple IFFT, filtering + interpolation, and compressed sensing, 
% and compressed sensing + filtering + interpolation

% Simple IFFT
% Get the good/bad data for the slice
[channel1, channel2, channel3] = get_data("1", 0);
% IFFT with no filtering
raw_image = get_image(channel1, channel2, channel3);
% Do some post processing on the image
raw_adj_img = adjust_image(raw_image);

% Filtering and interpolation 
% Pad the edges of the k-space with 0 (interpolation)
pad_channel1 = pad_channel(20, 80, channel1);
pad_channel2 = pad_channel(20, 80, channel2);
pad_channel3 = pad_channel(20, 80, channel3);
% Apply a smoothing window (effectively a low pass filter). Either hamming or tukeywin
w = get_window("tukeywin", .9, size(pad_channel1));
smoothed_channel1 = smooth_channel(pad_channel1, w);
smoothed_channel2 = smooth_channel(pad_channel2, w);
smoothed_channel3 = smooth_channel(pad_channel3, w);
% IFFT of filtered k-space data
filtered_img = get_image(smoothed_channel1, smoothed_channel2, smoothed_channel3);
% Post processing 
filtered_adj_img = adjust_image(filtered_img);

% Compressed sensing
% Take a subsampling of y to perform compressed sensing
%[n, C, y] = get_sparse_measurement(channel1, 10, "radial", [30,50]);
[n, C, y] = get_sparse_measurement(channel1, 500, "center", []);
%[n, C, y] = get_sparse_measurement(channel1, 10, "spiral", [300,100000,25]);
%[n, C, y] = get_sparse_measurement(channel1, 10, "sparse", [500]);

% Solve convex optimization problem for compressed sensing to get the
% sparse vector s
s = compressed_sensing(n, C, y, 50);
s = reshape(s, size(channel1,1), size(channel1,2));
filtered_s = get_image(s, s, s);
filtered_adj_s = adjust_image(filtered_s);

% Compressed sensing + filtering + interpolation
pad_s = pad_channel(20, 80, s);
w = get_window("tukeywin", .9, size(pad_s));
smoothed_s = smooth_channel(pad_s, w);
filtered_s2 = get_image(smoothed_s, smoothed_s, smoothed_s);
filtered_adj_s2 = adjust_image(filtered_s2);


%% Plot the images

close all
figure(1); 
axis image, 
colormap gray;
axis off

subplot(2,2,1)
imagesc(raw_adj_img(:,:,1));
subplot(2,2,2)
imagesc(filtered_adj_img(:,:,1));
subplot(2,2,3)
imagesc(filtered_adj_s(:,:,1));
subplot(2,2,4)
imagesc(filtered_adj_s(:,:,1));

%% Plot the k-space data
% Spatial frequency observations
figure(2); 
xlabel('Horizontal frequency bins')
ylabel('Vertical frequency bins');
subplot(2,2,1)
imagesc(100*log(abs(channel1)));
subplot(2,2,2)
imagesc(100*log(abs(smoothed_channel1)));
subplot(2,2,3)
imagesc(100*log(abs(s)));
subplot(2,2,4)
imagesc(100*log(abs(smoothed_s)));

figure(4); 
subplot(1,2,1)
imagesc(w);
subplot(1,2,2)
imagesc(C);

%% Functions

% Loads the data from slice slice_num
% If good_n_bad is 1, it will load the good data. Otherwise it'll load the
% bad data. The three channels are stored in [channel1, channel2, channel3]
function [channel1, channel2, channel3] = get_data(slice_num, good_n_bad)

    if (good_n_bad == 1)
        good_str = 'Good';
    else 
        good_str = 'Bad';
    end
    load(strcat('MRI_datasets/Slice',slice_num,'/',good_str,'Data/slice',slice_num,'_channel1.mat'));
    load(strcat('MRI_datasets/Slice',slice_num,'/',good_str,'Data/slice',slice_num,'_channel2.mat'));
    load(strcat('MRI_datasets/Slice',slice_num,'/',good_str,'Data/slice',slice_num,'_channel3.mat'));
    
    if (good_n_bad == 1)
        good_str = 'good';
    else 
        good_str = 'bad';
    end
    channel1 = eval(strcat('slice',slice_num,'_channel1_',good_str,'Data'));
    channel2 = eval(strcat('slice',slice_num,'_channel2_',good_str,'Data'));
    channel3 = eval(strcat('slice',slice_num,'_channel3_',good_str,'Data'));

end

function [n, C, y] = get_sparse_measurement(chan, n_over_p, psf, psf_settings);
    n = size(chan,1) * size(chan,2);
    kspace = reshape(chan, n, 1);
    p = round(n/(n_over_p));
    if (psf == "radial")
        C = radial_matrix(p,n,psf_settings(1),psf_settings(2));
    elseif (psf == "center")
        C = center_matrix(p,n);
    elseif (psf == "spiral")
        C = spiral_matrix(p,n,psf_settings(1), psf_settings(2), psf_settings(3));
    elseif (psf == "sparse")
        C = sparse_matrix(p,n,p*psf_settings(1));
    end

    y = C*kspace;
end

function s = compressed_sensing(n, C, y, error)
    C = double(C);
    y = double(y);
    cvx_begin;
        variable s(n) complex;
        minimize(norm(s,1));
        subject to 
            norm(C*s - y, 2) < error;
    cvx_end;
end

% Pad the k-space data with zeroes
function padchan = pad_channel(x_pad, y_pad, chan)

    y_dim = size(chan,1);
    x_dim = size(chan,2);
    x_padding = zeros(y_dim, x_pad);
    y_padding = zeros (y_pad, x_dim+x_pad*2);

    padchan = [x_padding, chan, x_padding];
    padchan = [y_padding; padchan; y_padding];
end

% Apply a smoothing window to the data. Supports hamming and tukey windows
% alg: "hamming" or "tukeywin"
% alg_tune: for hamming, this can be "periodic" or "symmetric". For
% tukeywin, choose a value between 0 and 1. For a value of 1, it's
% effectively a hamming window. 
function w = get_window(alg, alg_tune, dims)
    y_dim = dims(1);
    x_dim = dims(2);

    if (alg == "hamming")
        w = hamming(y_dim, alg_tune)*hamming(x_dim, alg_tune)';
    elseif (alg == "tukeywin")
        w = tukeywin(y_dim, alg_tune)*tukeywin(x_dim, alg_tune)';
    end
end

function smooth_chan = smooth_channel(chan, w)
    smooth_chan = chan.*w;
end

% Obtain the image from the k-space data using IFFT
function img = get_image(chan1, chan2, chan3)
    img(:,:,1) = ifftshift(ifft2(chan1),1);
    img(:,:,2) = ifftshift(ifft2(chan2),1);
    img(:,:,3) = ifftshift(ifft2(chan3),1);
end

% Do some post processing on the image
function adj_img = adjust_image(img)
    % clear compensation, preparation, based on fourier transformed blinked 
    % k-space data (Data_raw)
    clear_comp = linspace(10,0.1,size(img,2)).^2; 
    clear_matrix = repmat(clear_comp,[size(img,1) 1]);

    % combine 3 channels sum of squares and add clear compensation
    adj_img  = sqrt( abs(squeeze(img(:,:,1))).^2 + ...
                     abs(squeeze(img(:,:,2))).^2 + ...
                     abs(squeeze(img(:,:,3))).^2).* clear_matrix;  
    
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
    [aa, val] = hist(adj_img(:),linspace(0,max(...
                                      adj_img(:)),1000));
        thresh = val(find(cumsum(aa)/sum(aa) > std_within,1,'first'));
    
    % set threshold value to 65536
    adj_img = uint16(adj_img * 65536 / thresh); 
end
 
%% PSF Functions 

function C = center_matrix(rows, cols)
    num_cells = rows*cols;
    C = zeros(rows, cols);
    square_factor = cols/rows;

    for i = 1:cols
        for j = 1:rows
            distance_from_center = sqrt((square_factor*(j-rows/2))^2 + (i-cols/2)^2);
            max_distance = sqrt((square_factor*(rows/2))^2 + (cols/2)^2);
            normalized_distance = distance_from_center/max_distance;
            rand_num = rand/64;

            if (normalized_distance^2*rand < rand_num)
                C(j, i) = 1;
            end
        end 
    end
 end

 function C = radial_matrix(rows, cols, n, e)
    num_cells = rows*cols;
    C = zeros(rows, cols);
   
    % use row = m*col to create radial lines coming from the center
    m = linspace(-1,1,n)
    center_row = rows/2;
    center_col = cols/2;
    square_factor = cols/rows;

    for i = 1:cols
        for j = 1:rows
            for slope_idx = 1:n
                diff1 = abs(m(slope_idx)*(i-center_col) - (j-center_row)*square_factor);
                diff2 = abs(m(slope_idx)*(j-center_row)*square_factor - (i-center_col));
                if (min(diff1,diff2) <= e)
                    C(j, i) = 1;
                    break;
                end
            end
        end 
    end
 end

 function C = spiral_matrix(rows, cols, a, num_steps, num_spirals)
    C = zeros(rows, cols);
   
    center_row = rows/2;
    center_col = cols/2;
    square_factor = cols/rows;

    for spiral = 0:num_spirals
        beta = spiral*2*pi/num_spirals;
        for step = 0:num_steps
            % use r = a*theta, x = r*cos(theta), y = r*sin(theta)
            theta = step*4*pi/num_steps;
            r = a*(theta);
            i = round(r*cos(theta+beta)*square_factor + center_col);
            j = round(r*sin(theta+beta) + center_row);
            if (j < 2 || j > rows-1 || i < 2 || i > cols-1) 
                continue;
            end
            C(j, i) = 1;
            C(j+1, i) = 1;
            C(j-1, i) = 1;
            C(j, i+1) = 1;
            C(j, i-1) = 1;
        end
    end
 end

 function v=shuffle(v)
     v=v(randperm(length(v)));
 end

 function C = sparse_matrix(rows, cols, nonzero_vals)
    num_cells = rows*cols;
    remaining_cells = num_cells;
    C = zeros(rows, cols);

    for idx = shuffle([1:num_cells])

        if (nonzero_vals == 0) 
            return;
        end

        chance_of_1 = nonzero_vals/remaining_cells;
        rand_val = rand;
        if (rand_val < chance_of_1)
            C(idx) = 1;
            nonzero_vals = nonzero_vals - 1;
        else
            C(idx) = 0;
        end

        remaining_cells = remaining_cells - 1;
    end
 end

