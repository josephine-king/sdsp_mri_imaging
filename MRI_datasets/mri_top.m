%% Define which image we are looking at and load the data
functions = common_functions;
image_idx = "1";
[good_channel1, good_channel2, good_channel3] = functions.get_data(image_idx, 1);
[bad_channel1, bad_channel2, bad_channel3] = functions.get_data(image_idx, 0);

corrupted_lines = functions.get_corrupted_lines(bad_channel1);
corrupted_pixels1 = functions.get_corrupted_pixels(bad_channel1, corrupted_lines, .7);
corrupted_pixels2 = functions.get_corrupted_pixels(bad_channel2, corrupted_lines, .7);
corrupted_pixels3 = functions.get_corrupted_pixels(bad_channel3, corrupted_lines, .7);

%% Channel fusion
% Fuse channels 
noise_fused_good_channels = functions.fuse_channels_noise(good_channel1, good_channel2, good_channel3);
snr_fused_good_channels = functions.fuse_channels_snr(good_channel1, good_channel2, good_channel3);
average_fused_good_channels = functions.fuse_channels_average(good_channel1, good_channel2, good_channel3);
wiener_fused_good_channels = functions.fuse_channels_wiener(good_channel1, good_channel2, good_channel3);

% Get the images by applying IFFT
good_img = functions.get_image_non_fused(good_channel1, good_channel2, good_channel3);
good_adj_img = functions.adjust_image(good_img, 0);
[noise_fused_good_adj_img, MSE] = functions.get_image_and_mse(noise_fused_good_channels, good_adj_img);
[snr_fused_good_adj_img, MSE] = functions.get_image_and_mse(snr_fused_good_channels, good_adj_img);
[average_fused_good_adj_img, MSE] = functions.get_image_and_mse(average_fused_good_channels, good_adj_img);
[wiener_fused_good_adj_img, MSE] = functions.get_image_and_mse(wiener_fused_good_channels, good_adj_img);

% Get the images for the unfused channels
[unfused_channel1_img, unfused_MSE1] = functions.get_image_and_mse(good_channel1, good_adj_img);
[unfused_channel2_img, unfused_MSE2] = functions.get_image_and_mse(good_channel2, good_adj_img);
[unfused_channel3_img, unfused_MSE3] = functions.get_image_and_mse(good_channel3, good_adj_img);

figure(1)
axis image, 
colormap gray;
axis off
subplot(1,2,1)
imagesc(good_adj_img);
title("Good image, fusion after IFFT")
subplot(1,2,2)
imagesc(average_fused_good_adj_img);
title("Average fusion before IFFT")

figure(2)
axis image, 
colormap gray;
axis off
subplot(1,3,1)
imagesc(wiener_fused_good_adj_img); 
title("Wiener fusion using average approximation before IFFT")
subplot(1,3,2)
imagesc(noise_fused_good_adj_img);
title("Wiener fusion using noise approximation before IFFT")
subplot(1,3,3)
imagesc(snr_fused_good_adj_img);
title("Fusion using SNR")

figure(3)
axis image, 
colormap gray;
axis off
subplot(1,3,1)
imagesc(unfused_channel1_img); 
title("Unfused channel 1")
subplot(1,3,2)
imagesc(unfused_channel2_img);
title("Unfused channel 2")
subplot(1,3,3)
imagesc(unfused_channel3_img);
title("Unfused channel 3")

%% Simple filters
% Remove the corrupted lines
zeroed_channel1 = functions.zero_fill_corrupted_pixels(bad_channel1, corrupted_pixels1);
zeroed_channel2 = functions.zero_fill_corrupted_pixels(bad_channel2, corrupted_pixels1);
zeroed_channel3 = functions.zero_fill_corrupted_pixels(bad_channel3, corrupted_pixels1);
zeroed_channels = functions.fuse_channels_wiener(zeroed_channel1,zeroed_channel2,zeroed_channel3);

% Zero-fill the corrupted lines
removed_channel1 = functions.remove_corrupted_lines(bad_channel1, corrupted_lines);
removed_channel2 = functions.remove_corrupted_lines(bad_channel2, corrupted_lines);
removed_channel3 = functions.remove_corrupted_lines(bad_channel3, corrupted_lines);
removed_channels = functions.fuse_channels_wiener(removed_channel1,removed_channel2,removed_channel3);

% Average neighboring pixels
average_filtered_channel1 = functions.average_filter(bad_channel1, corrupted_pixels1);
average_filtered_channel2 = functions.average_filter(bad_channel2, corrupted_pixels2);
average_filtered_channel3 = functions.average_filter(bad_channel3, corrupted_pixels3);
average_filtered_channels = functions.fuse_channels_wiener(average_filtered_channel1, average_filtered_channel2, average_filtered_channel3);

% Replace with neighboring pixels
neighbor_replace_channel1 = functions.replace_corrupted_pixel_with_adjacent_pixel(bad_channel1, corrupted_pixels1,-1);
neighbor_replace_channel2 = functions.replace_corrupted_pixel_with_adjacent_pixel(bad_channel2, corrupted_pixels2,-1);
neighbor_replace_channel3 = functions.replace_corrupted_pixel_with_adjacent_pixel(bad_channel3, corrupted_pixels3,-1);
neighbor_replace_channels = functions.fuse_channels_wiener(neighbor_replace_channel1, neighbor_replace_channel2, neighbor_replace_channel3);

% Zero padding 
padded_channels = functions.pad_channel(128, 512, neighbor_replace_channels);
% Smoothing filter
w = functions.get_window("tukeywin", [.4,.4], size(neighbor_replace_channels));
smoothed_channels = neighbor_replace_channels.*w;
% Smoothing and padding
smoothed_padded_channels = functions.pad_channel(64, 256, smoothed_channels);

% Removed corrupted lines
removed_adj_img = functions.get_image_no_mse(removed_channels);
% Zeroed corrupted lines
zeroed_adj_img = functions.get_image_no_mse(zeroed_channels);
% Average filtered image:
[average_adj_img, average_img_MSE] = functions.get_image_and_mse(average_filtered_channels,good_adj_img);
% Replace with neighboring pixels:
[neighbor_replace_img, neighbor_replace_MSE] = functions.get_image_and_mse(neighbor_replace_channels,good_adj_img);

% Smoothed/lowpass
smoothed_adj_img = functions.get_image_no_mse(smoothed_channels);
% Padded
padded_adj_img = functions.get_image_no_mse(padded_channels);
% Smoothed and padded
smoothed_padded_adj_img = functions.get_image_no_mse(smoothed_padded_channels);

figure(1)
axis image, 
colormap gray;
axis off
subplot(1,4,1)
imagesc(removed_adj_img);
title("Corrupted lines removed")
subplot(1,4,2)
imagesc(zeroed_adj_img);
title("Corrupted pixels zeroed")
subplot(1,4,3)
imagesc(average_adj_img);
title("Neighbor averaging")
subplot(1,4,4)
imagesc(neighbor_replace_img);
title("Neighbor replacement")

figure(2); 
subplot(1,4,1)
imagesc(100*log(abs(removed_channels)));
title("K-space with corrupted lines removed")
subplot(1,4,2)
imagesc(100*log(abs(zeroed_channels)));
title("K-space with corrupted pixels zeroed")
subplot(1,4,3)
imagesc(100*log(abs(average_filtered_channels)));
title("K-space with neighbor averaging")
subplot(1,4,4)
imagesc(100*log(abs(neighbor_replace_channels)));
title("K-space with neighbor replacement")

figure(3)
subplot(1,4,1)
imagesc(100*log(abs(neighbor_replace_channels)));
title("Neighbor replacement no smoothing or padding")
subplot(1,4,2)
imagesc(100*log(abs(smoothed_channels)));
title("Neighbor replacement with smoothing")
subplot(1,4,3)
imagesc(100*log(abs(padded_channels)));
title("Neighbor replacement with padding")
subplot(1,4,4)
imagesc(100*log(abs(padded_channels)));
title("Neighbor replacement with smoothing and padding")

figure(4)
axis image, 
colormap gray;
axis off
subplot(1,4,1)
imagesc(neighbor_replace_img);
title("Neighbor replacement no smoothing or padding")
subplot(1,4,2)
imagesc(smoothed_adj_img);
title("Neighbor replacement with smoothing")
subplot(1,4,3)
imagesc(padded_adj_img);
title("Neighbor replacement with padding")
subplot(1,4,4)
imagesc(smoothed_padded_adj_img);
title("Neighbor replacement with smoothing and padding")

%% Comparing different kinds of Wiener filter
window_dims = [1,3];

wiener_filtered_channels = mri_wiener(bad_channel1, bad_channel2, bad_channel3, corrupted_pixels1, corrupted_pixels2, corrupted_pixels3, 0, "wiener", window_dims, "wiener", []);
piecewise_filtered_channels = mri_wiener(bad_channel1, bad_channel2, bad_channel3, corrupted_pixels1, corrupted_pixels2, corrupted_pixels3, 0, "wiener", window_dims, "piecewise", [16,4]);
center_filtered_channels = mri_wiener(bad_channel1, bad_channel2, bad_channel3, corrupted_pixels1, corrupted_pixels2, corrupted_pixels3, 0, "wiener", window_dims, "center_piecewise", [128,64]);

% Wiener filtered image:
[wiener_filtered_adj_img, filtered_img_MSE] = functions.get_image_and_mse(wiener_filtered_channels,good_adj_img);
% Wiener piecewise filtered image:
[piecewise_filtered_adj_img, piecewise_filtered_img_MSE] = functions.get_image_and_mse(piecewise_filtered_channels,good_adj_img);
% Wiener center filtered image:
[center_filtered_adj_img, center_filtered_img_MSE] = functions.get_image_and_mse(center_filtered_channels,good_adj_img);

figure(10)
axis image, 
colormap gray;
axis off
subplot(1,3,1)
imagesc(wiener_filtered_adj_img);
title("Wiener-filtered image")
subplot(1,3,2)
imagesc(piecewise_filtered_adj_img);
title("Piecewise Wiener-filtered image")
subplot(1,3,3)
imagesc(center_filtered_adj_img);
title("Center piecewise Wiener-filtered image")

figure(8); 
subplot(1,3,1)
imagesc(100*log(abs(wiener_filtered_channels)));
title("Wiener-filtered k-space data")
subplot(1,3,2)
imagesc(100*log(abs(piecewise_filtered_channels)));
title("Piecewise Wiener-filtered k-space data")
subplot(1,3,3)
imagesc(100*log(abs(center_filtered_channels)));
title("Center piecewise Wiener-filtered k-space data")


%% Comparing different methods - fuse first vs fuse second
window_dims = [1,3];

fused_bad_channels = functions.fuse_channels_wiener(bad_channel1, bad_channel2, bad_channel3);
corrupted_pixels = functions.get_corrupted_pixels(fused_bad_channels, corrupted_lines, -.6);

filtered_channels_fuse_first = mri_wiener(bad_channel1, bad_channel2, bad_channel3, corrupted_pixels, corrupted_pixels, corrupted_pixels, 1, "wiener", window_dims, "wiener", []);
filtered_channels_fuse_second = mri_wiener(bad_channel1, bad_channel2, bad_channel3, corrupted_pixels1, corrupted_pixels2, corrupted_pixels3, 0, "wiener", window_dims, "wiener", []);

% Wiener filtered image (fuse first):
[filtered_fuse_first_adj_img, wiener_filtered_fuse_first_MSE] = functions.get_image_and_mse(filtered_channels_fuse_first,good_adj_img);
% Wiener filtered image (fuse second):
[filtered_fuse_second_adj_img, wiener_filtered_fuse_second_MSE] = functions.get_image_and_mse(filtered_channels_fuse_second,good_adj_img);

figure(1)
axis image, 
colormap gray;
axis off
subplot(1,2,1)
imagesc(filtered_fuse_first_adj_img);
title("Wiener filtered image, fusion before filtering")
subplot(1,2,2)
imagesc(filtered_fuse_second_adj_img);
title("Wiener filtered image, fusion after filtering")

figure(2)
subplot(1,2,1)
imagesc(100*log(abs(filtered_channels_fuse_first)));
title("Wiener filtered kspace, fusion before filtering")
subplot(1,2,2)
imagesc(100*log(abs(filtered_channels_fuse_second)));
title("Wiener filtered kspace, fusion after filtering")

%% Comparing different window sizes for Wiener filtering
window_dims1 = [1,3];
window_dims2 = [3,3];
window_dims3 = [5,5];

[Rx1, rdx1] = functions.get_Rx_rdx(bad_channel1, window_dims1, corrupted_pixels1, 1, size(bad_channel1), 1);
[Rx2, rdx2] = functions.get_Rx_rdx(bad_channel1, window_dims2, corrupted_pixels1, 1, size(bad_channel1), 1);
[Rx3, rdx3] = functions.get_Rx_rdx(bad_channel1, window_dims3, corrupted_pixels1, 1, size(bad_channel1), 1);

filtered_channels_win_1_3 = mri_wiener(bad_channel1, bad_channel2, bad_channel3, corrupted_pixels1, corrupted_pixels2, corrupted_pixels3, 0, "wiener", window_dims1, "wiener", []);
filtered_channels_win_3_3 = mri_wiener(bad_channel1, bad_channel2, bad_channel3, corrupted_pixels1, corrupted_pixels2, corrupted_pixels3, 0, "wiener", window_dims2, "wiener", []);
filtered_channels_win_5_5 = mri_wiener(bad_channel1, bad_channel2, bad_channel3, corrupted_pixels1, corrupted_pixels2, corrupted_pixels3, 0, "wiener", window_dims3, "wiener", []);

% Wiener filtered image, 1 by 3 window:
[filtered_win_1_3_image, filtered_1_3_MSE] = functions.get_image_and_mse(filtered_channels_win_1_3,good_adj_img);
% Wiener filtered image, 3 by 3 window:
[filtered_win_3_3_image, filtered_3_3_MSE] = functions.get_image_and_mse(filtered_channels_win_3_3,good_adj_img);
% Wiener filtered image, 5 by 5 window:
[filtered_win_5_5_image, filtered_5_5_MSE] = functions.get_image_and_mse(filtered_channels_win_5_5,good_adj_img);

figure(1)
axis image, 
colormap gray;
axis off
subplot(1,3,1)
imagesc(filtered_win_1_3_image);
title("Wiener filtered image, 1 by 3 window")
subplot(1,3,2)
imagesc(filtered_win_3_3_image);
title("Wiener filtered image, 3 by 3 window")
subplot(1,3,3)
imagesc(filtered_win_5_5_image);
title("Wiener filtered image, 5 by 5 window")

figure(2)
subplot(1,3,1)
imagesc(100*log(abs(filtered_channels_win_1_3)));
title("Wiener filtered kspace, 1 by 3 window")
subplot(1,3,2)
imagesc(100*log(abs(filtered_channels_win_3_3)));
title("Wiener filtered kspace, 3 by 3 window")
subplot(1,3,3)
imagesc(100*log(abs(filtered_channels_win_5_5)));
title("Wiener filtered kspace, 5 by 5 window")

figure(3)
subplot(1,3,1)
imagesc(abs(Rx1));
title("Rx, 1 by 3 window")
subplot(1,3,2)
imagesc(abs(Rx2));
title("Rx, 3 by 3 window")
subplot(1,3,3)
imagesc(abs(Rx3));
title("Rx, 5 by 5 window")

figure(4)
subplot(1,3,1)
imagesc(abs(rdx1),[50, 250]);
title("Magnitude of rdx, 1 by 3 window")
colorbar
subplot(1,3,2)
imagesc(abs(rdx2),[50, 250]);
title("Magnitude of rdx, 3 by 3 window")
colorbar
subplot(1,3,3)
imagesc(abs(rdx3),[50, 250]);
title("Magnitude of rdx, 5 by 5 window")
colorbar

%% Getting corrupted pixels for different channels
[bad_channel_1_1, bad_channel_1_2, bad_channel_1_3] = functions.get_data("1", 0);
[bad_channel_2_1, bad_channel_2_2, bad_channel_2_3] = functions.get_data("2", 0);
[bad_channel_5_1, bad_channel_5_2, bad_channel_5_3] = functions.get_data("5", 0);

corrupted_pixels1 = functions.get_corrupted_pixels(bad_channel_1_1, corrupted_lines, .7);
corrupted_pixels2 = functions.get_corrupted_pixels(bad_channel_2_1, corrupted_lines, .7);
corrupted_pixels5 = functions.get_corrupted_pixels(bad_channel_5_1, corrupted_lines, .7);

figure(1)
subplot(2,3,1)
imagesc(100*log(abs(bad_channel_1_1)))
title("Scan 1, Channel 1")
subplot(2,3,2)
imagesc(100*log(abs(bad_channel_2_1)))
title("Scan 2, Channel 1")
subplot(2,3,3)
imagesc(100*log(abs(bad_channel_5_1)))
title("Scan 5, Channel 1")
subplot(2,3,4)
imagesc(corrupted_pixels1)
title("Identified corrupted pixels, Scan 1, Channel 1")
subplot(2,3,5)
imagesc(corrupted_pixels2)
title("Identified corrupted pixels, Scan 2, Channel 1")
subplot(2,3,6)
imagesc(corrupted_pixels5)
title("Identified corrupted pixels, Scan 5, Channel 1")

%% Conclusion - which techniques are the best?

bad_img = functions.get_image_non_fused(bad_channel1, bad_channel2, bad_channel3);
bad_adj_img = functions.adjust_image(bad_img, 0);

figure(1)
axis image, 
colormap gray;
axis off
subplot(1,4,1)
imagesc(bad_adj_img);
title("Original bad image with no correction")
subplot(1,4,2)
imagesc(filtered_fuse_second_adj_img);
title("Wiener filtered with 1-by-3 window")
subplot(1,4,3)
imagesc(neighbor_replace_img);
title("Filtered with neighbor replacement")
subplot(1,4,4)
imagesc(smoothed_padded_adj_img);
title("Filtered with neighbor replacement, smoothing, and padding")

%% Kalman filtering 
% We tried using a Kalman filter. However, our Kalman filter was basically
% equivalent to replacing corrupted pixels with the pixel to the left.
% Therefore we didn't think it added much value and we chose not to include
% it in our report

kalman_filtered_fuse_first_channels = mri_kalman(bad_channel1, bad_channel2, bad_channel3, corrupted_pixels1, corrupted_pixels2, corrupted_pixels3, 1, "wiener");
kalman_filtered_fuse_second_channels = mri_kalman(bad_channel1, bad_channel2, bad_channel3, corrupted_pixels1, corrupted_pixels2, corrupted_pixels3, 0, "wiener");

neighbor_replaced_channel1 = functions.replace_corrupted_pixel_with_adjacent_pixel(bad_channel1, corrupted_pixels1, -1);
neighbor_replaced_channel2 = functions.replace_corrupted_pixel_with_adjacent_pixel(bad_channel2, corrupted_pixels2, -1);
neighbor_replaced_channel3 = functions.replace_corrupted_pixel_with_adjacent_pixel(bad_channel3, corrupted_pixels3, -1);
neighbor_replaced_channels = functions.fuse_channels_wiener(neighbor_replaced_channel1, neighbor_replaced_channel2, neighbor_replaced_channel3);

% Unfiltered image
[bad_adj_img, bad_img_MSE] = functions.get_image_and_mse_nonfused(bad_channel1,bad_channel2,bad_channel3, good_adj_img);
% Kalman filtered image (fuse first):
[kalman_filtered_fuse_first_adj_img, kalman_filtered_fuse_first_MSE] = functions.get_image_and_mse(kalman_filtered_fuse_first_channels,good_adj_img);
% Kalman filtered image (fuse second):
[kalman_filtered_fuse_second_adj_img, kalman_filtered_fuse_second_MSE] = functions.get_image_and_mse(kalman_filtered_fuse_second_channels,good_adj_img);
% Neighbor replacement
[neighbor_replaced_adj_img, neighbor_replaced_MSE] = functions.get_image_and_mse(neighbor_replaced_channels,good_adj_img);

figure(1)
axis image, 
colormap gray;
axis off
subplot(1,3,1)
imagesc(bad_adj_img);
title("Original image")
subplot(1,3,2)
imagesc(kalman_filtered_fuse_first_adj_img);
title("Kalman filtered image, fusion before filtering")
subplot(1,3,3)
imagesc(kalman_filtered_fuse_second_adj_img);
title("Kalman filtered image, fusion after filtering")

figure(2)
subplot(1,3,1)
imagesc(100*log(abs(bad_channel1)));
title("Original kspace")
subplot(1,3,2)
imagesc(100*log(abs(kalman_filtered_fuse_first_channels)));
title("Kalman filtered kspace, fusion before filtering")
subplot(1,3,3)
imagesc(100*log(abs(kalman_filtered_fuse_second_channels)));
title("Kalman filtered kspace, fusion after filtering")

figure(3)
axis image, 
colormap gray;
axis off
subplot(1,2,1)
imagesc(neighbor_replaced_adj_img);
title("Filtered image from replacing corrupted pixels with left pixel")
subplot(1,2,2)
imagesc(kalman_filtered_fuse_second_adj_img);
title("Kalman filtered image, fusion after filtering")

figure(4)
subplot(1,2,1)
imagesc(100*log(abs(neighbor_replaced_channels)));
title("Filtered k-space from replacing corrupted pixels with left pixel")
subplot(1,2,2)
imagesc(100*log(abs(kalman_filtered_fuse_second_channels)));
title("Kalman filtered kspace, fusion after filtering")
