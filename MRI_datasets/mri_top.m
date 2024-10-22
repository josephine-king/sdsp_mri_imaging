%% Define which image we are looking at and load the data
functions = common_functions;
image_idx = "1";
[good_channel1, good_channel2, good_channel3] = functions.get_data(image_idx, 1);
[bad_channel1, bad_channel2, bad_channel3] = functions.get_data(image_idx, 0);

corrupted_lines1 = functions.get_corrupted_lines(bad_channel1, .9, 105);
corrupted_lines2 = functions.get_corrupted_lines(bad_channel2, .9, 105);
corrupted_lines3 = functions.get_corrupted_lines(bad_channel3, .9, 105);
corrupted_pixels1 = functions.get_corrupted_pixels(bad_channel1, corrupted_lines1, .7);
corrupted_pixels2 = functions.get_corrupted_pixels(bad_channel2, corrupted_lines2, .7);
corrupted_pixels3 = functions.get_corrupted_pixels(bad_channel3, corrupted_lines3, .7);

%% Channel fusion
% Fuse channels 
simple_fused_good_channels = functions.fuse_channels_simple(good_channel1, good_channel2, good_channel3);
wiener_fused_good_channels = functions.fuse_channels_wiener(good_channel1, good_channel2, good_channel3);

good_img = functions.get_image_non_fused(good_channel1, good_channel2, good_channel3);
good_adj_img = functions.adjust_image(good_img, 0);
[simple_fused_good_adj_img, MSE] = functions.get_image_and_mse(simple_fused_good_channels, good_adj_img);
[wiener_fused_good_adj_img, MSE] = functions.get_image_and_mse(wiener_fused_good_channels, good_adj_img);

bad_img = functions.get_image_non_fused(bad_channel1, bad_channel2, bad_channel3);
bad_adj_img = functions.adjust_image(bad_img, 0);

figure(1)
axis image, 
colormap gray;
axis off
subplot(1,3,1)
imagesc(good_adj_img);
title("Good image, fusion after IFFT")
subplot(1,3,2)
imagesc(simple_fused_good_adj_img); 
title("Good image, simple fusion before IFFT")
subplot(1,3,3)
imagesc(wiener_fused_good_adj_img);
title("Good image, Wiener fusion before IFFT")

figure(2)
subplot(2,3,1)
imagesc(100*log(abs(bad_channel1)));
title("K-space data for bad channel 1")
subplot(2,3,2)
imagesc(100*log(abs(bad_channel2)));
title("K-space data for bad channel 2")
subplot(2,3,3)
imagesc(100*log(abs(bad_channel3)));
title("K-space data for bad channel 3")
subplot(2,3,4)
imagesc(100*log(abs(good_channel1)));
title("K-space data for good channel 1")
subplot(2,3,5)
imagesc(100*log(abs(good_channel2)));
title("K-space data for good channel 2")
subplot(2,3,6)
imagesc(100*log(abs(good_channel3)));
title("K-space data for good channel 3")

figure(3)
axis image, 
colormap gray;
axis off
subplot(1,2,1)
imagesc(good_adj_img); 
title("Good image 1")
subplot(1,2,2)
imagesc(bad_adj_img);
title("Bad image 1")

figure(4)
subplot(1,2,1)
imagesc(100*log(abs(good_channel1))); 
title("Good k-space data, image 1")
subplot(1,2,2)
imagesc(100*log(abs(bad_channel1)));
title("Bad k-space data, image 1")

%% Simple filters
% Remove the corrupted lines
zeroed_channel1 = functions.remove_corrupted_lines(bad_channel1, corrupted_lines1, 1);
zeroed_channel2 = functions.remove_corrupted_lines(bad_channel2, corrupted_lines2, 1);
zeroed_channel3 = functions.remove_corrupted_lines(bad_channel3, corrupted_lines3, 1);
zeroed_channels = functions.fuse_channels_wiener(zeroed_channel1,zeroed_channel2,zeroed_channel3);

% Zero-fill the corrupted lines
removed_channel1 = functions.remove_corrupted_lines(bad_channel1, corrupted_lines1, 0);
removed_channel2 = functions.remove_corrupted_lines(bad_channel2, corrupted_lines2, 0);
removed_channel3 = functions.remove_corrupted_lines(bad_channel3, corrupted_lines3, 0);
removed_channels = functions.fuse_channels_wiener(removed_channel1,removed_channel2,removed_channel3);

% Low pass filter
w = functions.get_window("tukeywin", .9, size(bad_channel1));
smoothed_channel1 = bad_channel1.*w;
smoothed_channel2 = bad_channel2.*w;
smoothed_channel3 = bad_channel3.*w;
smoothed_channels = functions.fuse_channels_wiener(smoothed_channel1,smoothed_channel2,smoothed_channel3);

% Zero padding 
pad_channel1 = functions.pad_channel(64, 256, bad_channel1);
pad_channel2 = functions.pad_channel(64, 256, bad_channel2);
pad_channel3 = functions.pad_channel(64, 256, bad_channel3);
padded_channels = functions.fuse_channels_wiener(pad_channel1,pad_channel2,pad_channel3);

% Average neighboring pixels
average_filtered_channel1 = functions.average_filter(bad_channel1, corrupted_lines1);
average_filtered_channel2 = functions.average_filter(bad_channel2, corrupted_lines2);
average_filtered_channel3 = functions.average_filter(bad_channel3, corrupted_lines3);
average_filtered_channels = functions.fuse_channels_wiener(average_filtered_channel1, average_filtered_channel2, average_filtered_channel3);

% Removed corrupted lines
removed_adj_img = functions.get_image_no_mse(removed_channels);
% Zeroed corrupted lines
zeroed_adj_img = functions.get_image_no_mse(zeroed_channels);
% Smoothed/lowpass
smoothed_adj_img = functions.get_image_no_mse(smoothed_channels);
% Padded
padded_adj_img = functions.get_image_no_mse(padded_channels);
% Average filtered image:
[average_adj_img, average_img_MSE] = functions.get_image_and_mse(average_filtered_channels,good_adj_img);

figure(1)
axis image, 
colormap gray;
axis off
subplot(1,5,1)
imagesc(removed_adj_img);
title("Corrupted lines removed")
subplot(1,5,2)
imagesc(zeroed_adj_img);
title("Corrupted lines zeroed")
subplot(1,5,3)
imagesc(smoothed_adj_img);
title("Tukey smoothing window")
subplot(1,5,4)
imagesc(padded_adj_img);
title("K-space padding")
subplot(1,5,5)
imagesc(average_adj_img);
title("Average neighbors")

figure(2); 
subplot(1,5,1)
imagesc(100*log(abs(removed_channels)));
title("Corrupted lines removed")
subplot(1,5,2)
imagesc(100*log(abs(zeroed_channels)));
title("Corrupted lines zeroed")
subplot(1,5,3)
imagesc(100*log(abs(smoothed_channels)));
title("Tukey smoothing window")
subplot(1,5,4)
imagesc(100*log(abs(padded_channels)));
title("K-space padding")
subplot(1,5,5)
imagesc(100*log(abs(average_filtered_channels)));
title("Average neighbors")

%% Fuse first, Wiener filter second
window_dims = [3,3];

fused_bad_channels = functions.fuse_channels_wiener(bad_channel1, bad_channel2, bad_channel3);
corrupted_lines = functions.get_corrupted_lines(fused_bad_channels, 0.9, 105);
corrupted_pixels = functions.get_corrupted_pixels(fused_bad_channels, corrupted_lines, -.6);

[Rx, r_dx] = functions.get_Rx_rdx(fused_bad_channels, window_dims, corrupted_pixels, 1, size(fused_bad_channels), 1);
filtered_channels = mri_wiener(bad_channel1, bad_channel2, bad_channel3, corrupted_pixels, corrupted_pixels, corrupted_pixels, 1, "wiener", window_dims, "wiener", []);
piecewise_filtered_channels = mri_wiener(bad_channel1, bad_channel2, bad_channel3, corrupted_pixels, corrupted_pixels, corrupted_pixels, 1, "wiener", window_dims, "piecewise", [16,4]);
center_filtered_channels = mri_wiener(bad_channel1, bad_channel2, bad_channel3, corrupted_pixels, corrupted_pixels, corrupted_pixels, 1, "wiener", window_dims, "center_piecewise", [128,64]);

%% Wiener filter first, fuse second
% Find the corrupted lines and replace them with the average of the lines
% next to them
window_dims = [1,3];

[Rx1, r_dx1] = functions.get_Rx_rdx(bad_channel1, window_dims, corrupted_pixels1, 1, size(fused_bad_channels), 1);
[Rx2, r_dx2] = functions.get_Rx_rdx(bad_channel2, window_dims, corrupted_pixels2, 1, size(fused_bad_channels), 1);
[Rx3, r_dx3] = functions.get_Rx_rdx(bad_channel3, window_dims, corrupted_pixels3, 1, size(fused_bad_channels), 1);

wiener_filtered_channels = mri_wiener(bad_channel1, bad_channel2, bad_channel3, corrupted_pixels1, corrupted_pixels2, corrupted_pixels3, 0, "wiener", window_dims, "wiener", []);
piecewise_filtered_channels = mri_wiener(bad_channel1, bad_channel2, bad_channel3, corrupted_pixels1, corrupted_pixels2, corrupted_pixels3, 0, "wiener", window_dims, "piecewise", [16,4]);
center_filtered_channels = mri_wiener(bad_channel1, bad_channel2, bad_channel3, corrupted_pixels1, corrupted_pixels2, corrupted_pixels3, 0, "wiener", window_dims, "center_piecewise", [128,64]);

%% Plotting
% Unfiltered image
[bad_adj_img, bad_img_MSE] = functions.get_image_and_mse_nonfused(bad_channel1,bad_channel2,bad_channel3, good_adj_img);
% Wiener filtered image:
[wiener_filtered_adj_img, filtered_img_MSE] = functions.get_image_and_mse(wiener_filtered_channels,good_adj_img);
% Wiener piecewise filtered image:
[piecewise_filtered_adj_img, piecewise_filtered_img_MSE] = functions.get_image_and_mse(piecewise_filtered_channels,good_adj_img);
% Wiener center filtered image:
[center_filtered_adj_img, center_filtered_img_MSE] = functions.get_image_and_mse(center_filtered_channels,good_adj_img);

MSEs = [bad_img_MSE, average_img_MSE, filtered_img_MSE, piecewise_filtered_img_MSE, center_filtered_img_MSE];

figure(10)
axis image, 
colormap gray;
axis off
subplot(2,2,1)
imagesc(bad_adj_img);
title("Original image")
subplot(2,2,2)
imagesc(wiener_filtered_adj_img);
title("Wiener filtered image")
subplot(2,2,3)
imagesc(piecewise_filtered_adj_img);
title("Piecewise Wiener filtered image")
subplot(2,2,4)
imagesc(center_filtered_adj_img);
title("Center Wiener filtered image")

figure(8); 
subplot(2,2,1)
imagesc(100*log(abs(average_filtered_channels)));
title("Fused + cleaned k-space data")
subplot(2,2,2)
imagesc(100*log(abs(wiener_filtered_channels)));
title("Fused + Wiener filtered k-space data")
subplot(2,2,3)
imagesc(100*log(abs(piecewise_filtered_channels)));
title("Fused + Wiener piecewise filtered k-space data")
subplot(2,2,4)
imagesc(100*log(abs(center_filtered_channels)));
title("Fused + Wiener center filtered k-space data")

figure(4)
bar(MSEs)

figure(7)
subplot(2,3,1)
imagesc(100*log(abs(bad_channel1)))
subplot(2,3,2)
imagesc(100*log(abs(bad_channel2)))
subplot(2,3,3)
imagesc(100*log(abs(bad_channel3)))
subplot(2,3,4)
imagesc(corrupted_pixels1)
subplot(2,3,5)
imagesc(corrupted_pixels2)
subplot(2,3,6)
imagesc(corrupted_pixels3)

%% Kalman filtering 

kalman_dims = [1,3];

kalman_filtered_fuse_first_channels = mri_kalman(bad_channel1, bad_channel2, bad_channel3, corrupted_pixels1, corrupted_pixels2, corrupted_pixels3, 1, "wiener", kalman_dims);
kalman_filtered_fuse_second_channels = mri_kalman(bad_channel1, bad_channel2, bad_channel3, corrupted_pixels1, corrupted_pixels2, corrupted_pixels3, 0, "wiener", kalman_dims);

% Unfiltered image
[bad_adj_img, bad_img_MSE] = functions.get_image_and_mse_nonfused(bad_channel1,bad_channel2,bad_channel3, good_adj_img);
% Kalman filtered image (fuse first):
[kalman_filtered_fuse_first_adj_img, kalman_filtered_fuse_first_MSE] = functions.get_image_and_mse(kalman_filtered_fuse_first_channels,good_adj_img);
% Kalman filtered image (fuse second):
[kalman_filtered_fuse_second_adj_img, kalman_filtered_fuse_second_MSE] = functions.get_image_and_mse(kalman_filtered_fuse_second_channels,good_adj_img);

figure(3)
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

figure(1)
subplot(2,3,1)
imagesc(100*log(abs(bad_channel1)))
subplot(2,3,2)
imagesc(100*log(abs(bad_channel2)))
subplot(2,3,3)
imagesc(100*log(abs(bad_channel3)))
subplot(2,3,4)
imagesc(corrupted_pixels1)
subplot(2,3,5)
imagesc(corrupted_pixels2)
subplot(2,3,6)
imagesc(corrupted_pixels3)