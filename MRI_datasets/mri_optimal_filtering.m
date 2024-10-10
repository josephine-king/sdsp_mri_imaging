%%
[channel1, channel2, channel3] = get_data("1", 0);
[good_channel1, good_channel2, good_channel3] = get_data("1", 1);

M = motion_blur_psf(10, 30, size(good_channel1));
blurred_image = M.*good_channel1;

low_pass_filter = tukeywin(512,.4)*tukeywin(128,.4)';

% Calculate noise variance from part of image with no image pixels
% The top part of the image has no image pixels. Calculate the variance of
% that noise
[V1,var_v1,noise_img1] = calculate_noise_variance(channel1);
[V2,var_v2,noise_img2] = calculate_noise_variance(channel2);
[V3,var_v3,noise_img3] = calculate_noise_variance(channel3);
avg_var_v = (var_v1 + var_v2 + var_v3)/3;

% Fuse the channels
power1 = abs(channel1).^2;
power2 = abs(channel2).^2;
power3 = abs(channel3).^2;
signal_power = (P1+P2+P3)./3;
SNR1 = norm(power1,2)/var_v1;
SNR2 = norm(power2,2)/var_v2;
SNR3 = norm(power3,2)/var_v3;
avg_SNR = (SNR1+SNR2+SNR3)/3;

filtered_chan1 = channel1.*(power1./(power1+abs(V1).^2));
filtered_chan2 = channel2.*(power2./(power2+abs(V2).^2));
filtered_chan3 = channel3.*(power3./(power3+abs(V3).^2));
filtered_chan1(isnan(filtered_chan1)) = 0;
filtered_chan2(isnan(filtered_chan2)) = 0;
filtered_chan3(isnan(filtered_chan3)) = 0;

fused_channels = (filtered_chan1+filtered_chan2+filtered_chan3)/3;

% Find the corrupted lines and replace them with the average of the lines
% next to them
[D,C] = get_corrupted_lines(fused_channels, fused_channels, 2);
% Find the degradation filter
H = fused_channels./C;
H(isnan(H)) = 0;
H = H;
% Find the Wiener filter
W = conj(H)./(abs(H).^2 + 1/avg_SNR);
W(isnan(W)) = 0;
% Apply the Wiener filter to the data
F = W.*fused_channels;%.*low_pass_filter;
img = get_image(F,F,F);
adj_img = adjust_image(img);
non_img = get_image(channel1, channel2, channel3);
non_adj_img = adjust_image(non_img);
blurred_img = get_image(blurred_image, blurred_image, blurred_image);
blurred_adj_img = adjust_image(blurred_img);
good_img = get_image(good_channel1, good_channel2, good_channel3);
good_adj_img = adjust_image(good_img);

figure(1)
subplot(2,3,1)
imagesc(100*log(abs(channel1)));
title("K-space data for channel 1")
subplot(2,3,2)
imagesc(100*log(abs(channel2)));
title("K-space data for channel 2")
subplot(2,3,3)
imagesc(100*log(abs(channel3)));
title("K-space data for channel 3")
subplot(2,3,4)
imagesc(100*log(abs(good_channel1)));
title("K-space data for channel 1")
subplot(2,3,5)
imagesc(100*log(abs(good_channel2)));
title("K-space data for channel 2")
subplot(2,3,6)
imagesc(100*log(abs(good_channel3)));
title("K-space data for channel 3")

figure(2)
imagesc(abs(W));
title("Wiener filter")

figure(3)
axis image, 
colormap gray;
axis off
subplot(1,3,1)
imagesc(good_adj_img(:,:,1));
title("Original good image")
subplot(1,3,2)
imagesc(non_adj_img(:,:,1));
title("Original bad image")
subplot(1,3,3)
imagesc(adj_img(:,:,1));
title("Filtered bad image")

figure(4); 
subplot(1,2,1)
imagesc(100*log(abs(fused_channels)));
title("Fused k-space data (left)")
subplot(1,2,2)
imagesc(100*log(abs(F)));
title("Fused + filtered k-space data (right)")

figure(5)
axis image, 
colormap gray;
axis off
imagesc(blurred_adj_img(:,:,1));
title("Blurred image (left)")


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

function [D,C] = get_corrupted_lines(channel, filtered_channel, threshold)
    num_cols = size(channel,2);
    num_rows = size(channel,1);
    C = channel;
    D = zeros(num_rows,num_cols);
    for col = 3:num_cols-2
        this_col = filtered_channel(:,col);
        left_col = filtered_channel(:,col-1);
        right_col = filtered_channel(:,col+1);

        diff_left = norm((this_col - left_col),2);
        diff_right = norm((this_col - right_col),2);

        r = ((diff_left+diff_right)/2)/(norm(this_col,2));
        if (ismember(col, [8,22,38,54,70,86,102,118]))
            D(:,col) = ones(num_rows,1);
            C(:,col) = channel(:,col-1);% + channel(:,col+1))./2;
        end
    end
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
    [aa, val] = hist(adj_img(:),linspace(0,max(adj_img(:)),1000));
        thresh = val(find(cumsum(aa)/sum(aa) > std_within,1,'first'));
    
    % set threshold value to 65536
    adj_img = uint16(adj_img * 65536 / thresh); 
end

function [V,var_v,adj_img] = calculate_noise_variance(channel)
    img = get_image(channel,channel,channel);

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
    lower_y_bound = 1;
    upper_y_bound = 128;
    crop_y = [lower_y_bound : upper_y_bound]; % crop coordinates
    adj_img = adj_img(crop_y, :);
    adj_img = [adj_img;adj_img;adj_img;adj_img]

    var_v = var(double(adj_img(:)));
    V = fftshift(fft2(adj_img));
end


function M = motion_blur_psf(length, angle, image_size)
    % Create motion blur kernel using fspecial
    m = fspecial('motion', length, angle);

    % Compute the Fourier Transform of the motion blur kernel
    M = fftshift(fft2(m, image_size(1), image_size(2)));
    
end

