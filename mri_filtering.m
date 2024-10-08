%% Script to translate the K-spcae images into spatial eye images
clc; clear all; close all;

% Get the good/bad data for the slice
[channel1, channel2, channel3] = get_data("1", 1);
% IFFT with no filtering
raw_image = get_image(channel1, channel2, channel3);
% Do some post processing on the image
raw_adj_img = adjust_image(raw_image);

% Pad the edges of the k-space with 0 (interpolation)
[pad_channel1, pad_channel2, pad_channel3] = pad_channels(20, 80, channel1, channel2, channel3);
% Apply a smoothing window (effectively a low pass filter). Either hamming or tukeywin
[w, smoothed_channel1, smoothed_channel2, smoothed_channel3] = smooth_channels("tukeywin", .9, pad_channel1, pad_channel2, pad_channel3);
% IFFT of filtered k-space data
filtered_img = get_image(smoothed_channel1, smoothed_channel2, smoothed_channel3);
% Post processing 
filtered_adj_img = adjust_image(filtered_img);

% Plot the images
close all
figure(1); 
subplot(1,2,1)
imagesc(filtered_adj_img(:,:,1));
axis image, 
colormap gray;
axis off
subplot(1,2,2)
imagesc(raw_adj_img(:,:,1));
axis image, 
colormap gray;
axis off

% Spatial frequency observations
figure(3); 
imagesc(100*log(abs(smoothed_channel1)));

figure(4); 
imagesc(w);

xlabel('Horizontal frequency bins')
ylabel('Vertical frequency bins');


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

% Pad the k-space data with zeroes
function [padchan1, padchan2, padchan3] = pad_channels(x_pad, y_pad, chan1, chan2, chan3)
    dims = size(chan1);
    y_dim = dims(1)
    x_dim = dims(2)
    x_padding = zeros(y_dim, x_pad);
    y_padding = zeros (y_pad, x_dim+x_pad*2);

    padchan1 = [x_padding, chan1, x_padding];
    padchan1 = [y_padding; padchan1; y_padding];
    padchan2 = [x_padding, chan2, x_padding];
    padchan2 = [y_padding; padchan2; y_padding];
    padchan3 = [x_padding, chan3, x_padding];
    padchan3 = [y_padding; padchan3; y_padding];
end

% Apply a smoothing window to the data. Supports hamming and tukey windows
% alg: "hamming" or "tukeywin"
% alg_tune: for hamming, this can be "periodic" or "symmetric". For
% tukeywin, choose a value between 0 and 1. For a value of 1, it's
% effectively a hamming window. 
function [w, smoothchan1, smoothchan2, smoothchan3] = smooth_channels(alg, alg_tune, chan1, chan2, chan3)
    
    dims = size(chan1);
    y_dim = dims(1);
    x_dim = dims(2);

    if (alg == "hamming")
        w = hamming(y_dim, alg_tune)*hamming(x_dim, alg_tune)';
    elseif (alg == "tukeywin")
        w = tukeywin(y_dim, alg_tune)*tukeywin(x_dim, alg_tune)';
    end

    smoothchan1 = chan1.*w;
    smoothchan2 = chan2.*w;
    smoothchan3 = chan3.*w;

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
    lower_y_bound = (y_dim - x_dim)/2
    upper_y_bound = lower_y_bound + x_dim
    crop_y = [lower_y_bound : upper_y_bound]; % crop coordinates
    adj_img = adj_img(crop_y, :);
    adj_img_dims = size(adj_img)

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
