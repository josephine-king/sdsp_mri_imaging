%%
functions = common_functions;
image_idx = "5";
[good_channel1, good_channel2, good_channel3] = functions.get_data(image_idx, 1);
[bad_channel1, bad_channel2, bad_channel3] = functions.get_data(image_idx, 0);

% Fuse channels 
simple_fused_good_channels = functions.fuse_channels_simple(good_channel1, good_channel2, good_channel3);
wiener_fused_good_channels = functions.fuse_channels_wiener(good_channel1, good_channel2, good_channel3);

good_img = functions.get_image_non_fused(good_channel1, good_channel2, good_channel3);
good_adj_img = functions.adjust_image(good_img, 0);
[simple_fused_good_adj_img, MSE] = functions.get_image_and_mse(simple_fused_good_channels, good_adj_img);
[wiener_fused_good_adj_img, MSE] = functions.get_image_and_mse(wiener_fused_good_channels, good_adj_img);

figure(3)
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


%% Fuse first, clean second
% Find the corrupted lines and replace them with the average of the lines
% next to them
functions = common_functions;
image_idx = "4";
[bad_channel1, bad_channel2, bad_channel3] = functions.get_data(image_idx, 0);
fused_bad_channels = functions.fuse_channels_wiener(bad_channel1, bad_channel2, bad_channel3);

corrupted_lines = functions.get_corrupted_lines(fused_bad_channels, 0.9, 105);
corrupted_pixels = functions.get_corrupted_pixels(bad_channel1, corrupted_lines1, -.6);
cleaned_channels = clean_corrupted_pixels(fused_bad_channels, corrupted_pixels);

window_dims = [1,3];
[Rx, r_dx] = functions.get_Rx_rdx(fused_bad_channels, window_dims, corrupted_pixels, 1, size(fused_bad_channels), 1);
filtered_channels = mri_wiener(bad_channel1, bad_channel2, bad_channel3, corrupted_pixels, corrupted_pixels, corrupted_pixels, 1, "wiener", window_dims, "wiener", []);
piecewise_filtered_channels = mri_wiener(bad_channel1, bad_channel2, bad_channel3, corrupted_pixels, corrupted_pixels, corrupted_pixels, 1, "wiener", window_dims, "piecewise", [16,4]);
center_filtered_channels = mri_wiener(bad_channel1, bad_channel2, bad_channel3, corrupted_pixels, corrupted_pixels, corrupted_pixels, 1, "wiener", window_dims, "center_piecewise", [128,64]);

%% Clean first, fuse second
% Find the corrupted lines and replace them with the average of the lines
% next to them
functions = common_functions;
image_idx = "1";
[bad_channel1, bad_channel2, bad_channel3] = functions.get_data(image_idx, 0);
fused_bad_channels = functions.fuse_channels_simple(bad_channel1, bad_channel2, bad_channel3);

window_dims = [3,3];

corrupted_lines1 = functions.get_corrupted_lines(bad_channel1, .9, 105);
corrupted_lines2 = functions.get_corrupted_lines(bad_channel2, .9, 105);
corrupted_lines3 = functions.get_corrupted_lines(bad_channel3, .9, 105);
corrupted_pixels1 = functions.get_corrupted_pixels(bad_channel1, corrupted_lines1, 1.3);
corrupted_pixels2 = functions.get_corrupted_pixels(bad_channel2, corrupted_lines2, 1.3);
corrupted_pixels3 = functions.get_corrupted_pixels(bad_channel3, corrupted_lines3, 1.3);

cleaned_channel1 = clean_corrupted_pixels(bad_channel1, corrupted_pixels1);
cleaned_channel2 = clean_corrupted_pixels(bad_channel2, corrupted_pixels2);
cleaned_channel3 = clean_corrupted_pixels(bad_channel3, corrupted_pixels3);
cleaned_channels = functions.fuse_channels_wiener(cleaned_channel1, cleaned_channel2, cleaned_channel3);

[Rx1, r_dx1] = functions.get_Rx_rdx(bad_channel1, window_dims, corrupted_pixels1, 1, size(fused_bad_channels), 1);
[Rx2, r_dx2] = functions.get_Rx_rdx(bad_channel2, window_dims, corrupted_pixels2, 1, size(fused_bad_channels), 1);
[Rx3, r_dx3] = functions.get_Rx_rdx(bad_channel3, window_dims, corrupted_pixels3, 1, size(fused_bad_channels), 1);

filtered_channels = mri_wiener(bad_channel1, bad_channel2, bad_channel3, corrupted_pixels1, corrupted_pixels2, corrupted_pixels3, 0, "wiener", window_dims, "wiener", []);
piecewise_filtered_channels = mri_wiener(bad_channel1, bad_channel2, bad_channel3, corrupted_pixels1, corrupted_pixels2, corrupted_pixels3, 0, "wiener", window_dims, "piecewise", [16,4]);
center_filtered_channels = mri_wiener(bad_channel1, bad_channel2, bad_channel3, corrupted_pixels1, corrupted_pixels2, corrupted_pixels3, 0, "wiener", window_dims, "center_piecewise", [128,64]);


%%
% Unfiltered image
[bad_adj_img, bad_img_MSE] = functions.get_image_and_mse_nonfused(bad_channel1,bad_channel2,bad_channel3, good_adj_img);

% Average filtered image:
[cleaned_adj_img, cleaned_img_MSE] = functions.get_image_and_mse(cleaned_channels,good_adj_img);

% Wiener filtered image:
[filtered_adj_img, filtered_img_MSE] = functions.get_image_and_mse(filtered_channels,good_adj_img);

% Wiener piecewise filtered image:
[piecewise_filtered_adj_img, piecewise_filtered_img_MSE] = functions.get_image_and_mse(piecewise_filtered_channels,good_adj_img);

% Wiener center filtered image:
[center_filtered_adj_img, center_filtered_img_MSE] = functions.get_image_and_mse(center_filtered_channels,good_adj_img);

MSEs = [bad_img_MSE, cleaned_img_MSE, filtered_img_MSE, piecewise_filtered_img_MSE, center_filtered_img_MSE];

%% Plot images

figure(10)
axis image, 
colormap gray;
axis off
subplot(2,2,1)
imagesc(bad_adj_img);
title("Fused and cleaned image")
subplot(2,2,2)
imagesc(filtered_a ...
    dj_img);
title("Wiener filtered image")
subplot(2,2,3)
imagesc(piecewise_filtered_adj_img);
title("Piecewise Wiener filtered image")
subplot(2,2,4)
imagesc(center_filtered_adj_img);
title("Center Wiener filtered image")

figure(8); 
subplot(2,2,1)
imagesc(100*log(abs(cleaned_channels)));
title("Fused + cleaned k-space data")
subplot(2,2,2)
imagesc(100*log(abs(filtered_channels)));
title("Fused + Wiener filtered k-space data")
subplot(2,2,3)
imagesc(100*log(abs(piecewise_filtered_channels)));
title("Fused + Wiener piecewise filtered k-space data")
subplot(2,2,4)
imagesc(100*log(abs(center_filtered_channels)));
title("Fused + Wiener center filtered k-space data")

figure(4)
bar(MSEs)

figure(5)
subplot(1,2,1)
imagesc(abs(Rx),[0,2000])
subplot(1,2,2)
imagesc(abs(r_dx),[1000,1500])

figure(6)
subplot(2,3,1)
imagesc(abs(Rx1))
subplot(2,3,2)
imagesc(abs(Rx2))
subplot(2,3,3)
imagesc(abs(Rx3))
subplot(2,3,4)
imagesc(abs(r_dx1))
subplot(2,3,5)
imagesc(abs(r_dx2))
subplot(2,3,6)
imagesc(abs(r_dx3))

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

%% Functions
function filtered_channels = mri_wiener(channel1, channel2, channel3, cp1, cp2, cp3, fuse_first, fuse_type, window_dims, wiener_type, wiener_type_args)
    functions = common_functions;

    if (fuse_first == 1)
        if (fuse_type == "simple")
            channels = functions.fuse_channels_simple(channel1,channel2,channel3);
        else
            channels = functions.fuse_channels_wiener(channel1,channel2,channel3);
        end
        if (wiener_type == "wiener")
            filtered_channels = wiener_filter(channels, cp1, window_dims);
        elseif (wiener_type == "piecewise")
            filtered_channels = piecewise_wiener_filter(channels, cp1, window_dims, wiener_type_args(1), wiener_type_args(2));
        elseif (wiener_type == "center_piecewise")
            filtered_channels = center_piecewise_wiener_filter(channels, cp1, window_dims, [wiener_type_args(1) wiener_type_args(2)]);
        end
    else
        if (wiener_type == "wiener")
            filtered_channel1 = wiener_filter(channel1, cp1, window_dims);
            filtered_channel2 = wiener_filter(channel2, cp2, window_dims);
            filtered_channel3 = wiener_filter(channel3, cp3, window_dims);
        elseif (wiener_type == "piecewise")
            filtered_channel1 = piecewise_wiener_filter(channel1, cp1, window_dims, wiener_type_args(1), wiener_type_args(2));
            filtered_channel2 = piecewise_wiener_filter(channel2, cp2, window_dims, wiener_type_args(1), wiener_type_args(2));
            filtered_channel3 = piecewise_wiener_filter(channel3, cp3, window_dims, wiener_type_args(1), wiener_type_args(2));
        elseif (wiener_type == "center_piecewise")
            filtered_channel1 = center_piecewise_wiener_filter(channel1, cp1, window_dims, [wiener_type_args(1), wiener_type_args(2)]);
            filtered_channel2 = center_piecewise_wiener_filter(channel2, cp2, window_dims, [wiener_type_args(1), wiener_type_args(2)]);
            filtered_channel3 = center_piecewise_wiener_filter(channel3, cp3, window_dims, [wiener_type_args(1), wiener_type_args(2)]);
        end
        if (fuse_type == "simple")
            filtered_channels = functions.fuse_channels_simple(filtered_channel1,filtered_channel2,filtered_channel3);
        else
            filtered_channels = functions.fuse_channels_wiener(filtered_channel1,filtered_channel2,filtered_channel3);
        end
    end
end

function filtered_channel = center_piecewise_wiener_filter(channel, corrupted_pixels, window_dims, center_dims)
    functions = common_functions;
    filtered_channel = channel;
    % The square window excludes the middle column, so the order is smaller
    % than the window by a factor of window_size
    window_rows = window_dims(1);
    window_cols = window_dims(2);
    order = window_rows*window_cols - window_rows;
    row_padding = (window_dims(1)-1)/2;
    col_padding = (window_dims(2)-1)/2;

    center_start_rows = max(round((size(channel,1)-center_dims(1))/2),1);
    center_start_cols= max(round((size(channel,2)-center_dims(2))/2));
    center_end_rows = center_start_rows + center_dims(1);
    center_end_cols = center_start_cols + center_dims(2);

    % Calculate a separate Rx and r_dx for the center vs the edges
    [Rx_center, r_dx_center] = functions.get_Rx_rdx(channel, window_dims, corrupted_pixels, 1, center_dims, 1);
    [Rx_edge, r_dx_edge] = functions.get_Rx_rdx(channel, window_dims, corrupted_pixels, 0, center_dims, 1);

    % Calculate the optimal filter w
    w_center = inv(Rx_center)*r_dx_center;
    w_edge = inv(Rx_edge)*r_dx_edge;
    for row = 1+row_padding:size(channel,1)-row_padding
        for col = 1+col_padding:size(channel,2)-col_padding
            % Check if this is a corrupted pixel that needs correcting
            if (corrupted_pixels(row,col)==1)
                % Get the x vector by looping through the neighboring columns
                x = [];
                for x_col = [col-(col_padding):col-1, col+1:col+(col_padding)]
                    x = cat(1,x,channel(row-row_padding:row+row_padding, x_col));                        
                end
                if (row > center_start_rows && row < center_end_rows && col > center_start_cols && col < center_end_cols)
                    d = w_center'*x;
                else
                    d =w_edge'*x;
                end
                filtered_channel(row,col) = d;
            end
        end
    end
end

function filtered_channel = piecewise_wiener_filter(channel, corrupted_pixels, window_dims, num_rows, num_cols)
    filtered_channel = channel;
    % Round down. Any remaining pixels will be used in the last row/col
    pixels_per_row = floor(size(channel,1)/num_rows);
    pixels_per_col = floor(size(channel,2)/num_cols);

    order = window_dims(1)*window_dims(2) - window_dims(1);
    row_padding = (window_dims(1)-1)/2;
    col_padding = (window_dims(2)-1)/2;

    for row = 1:num_rows
        for col = 1:num_cols
            row_start = (row-1)*pixels_per_row+1;
            col_start = (col-1)*pixels_per_col+1;
            if (row == num_rows)
                row_end = size(channel,1);
            else
                row_end = row_start+pixels_per_row;
            end
            if (col == num_cols)
                col_end = size(channel,2);
            else
                col_end = col_start+pixels_per_col;
            end
            % Add some padding
            row_start_pad = max(row_start, row_start-row_padding);
            col_start_pad = max(col_start, col_start-col_padding);
            row_end_pad = min(row_end, row_end+row_padding);
            col_end_pad = min(col_end, col_end+col_padding);

            channel_chunk = channel(row_start_pad:row_end_pad, col_start_pad:col_end_pad);
            corrupted_pixel_chunk = corrupted_pixels(row_start_pad:row_end_pad, col_start_pad:col_end_pad);
            filtered_chunk = wiener_filter(channel_chunk, corrupted_pixel_chunk, window_dims);
            % Remove the padding from the filtered chunk
            if (row_start_pad ~= row_start) filtered_chunk(row_start_pad:row_start,:) = []; end
            if (row_end_pad ~= row_end) filtered_chunk(row_end:row_end_pad,:) = []; end
            if (col_start_pad ~= col_start) filtered_chunk(:,col_start_pad:col_start) = []; end
            if (col_end_pad ~= col_end) filtered_chunk(:,col_end:col_end_pad) = []; end

            filtered_channel(row_start:row_end, col_start:col_end) = filtered_chunk;
        end
    end
end

function filtered_channel = wiener_filter(channel, corrupted_pixels, window_dims)
    functions = common_functions;
    filtered_channel = channel;
    % The square window excludes the middle column, so the order is smaller
    % than the window by a factor of window_size
    row_padding = (window_dims(1)-1)/2;
    col_padding = (window_dims(2)-1)/2;
    % Find the correlation matrix and cross correlations between d and x
    [Rx, r_dx] = functions.get_Rx_rdx(channel, window_dims, corrupted_pixels, 1, size(channel), 1);
    % Calculate the optimal filter w
    w = inv(Rx)*r_dx;
    for row = 1+row_padding:size(channel,1)-row_padding
        for col = 1+col_padding:size(channel,2)-col_padding
            % Check if this is a corrupted pixel that needs correcting
            if (corrupted_pixels(row,col)==1)
                % Get the x vector by looping through the neighboring columns
                x = [];
                for x_col = [col-(col_padding):col-1, col+1:col+(col_padding)]
                    x = cat(1,x,channel(row-row_padding:row+row_padding, x_col));                        
                end
                d = w'*x;
                filtered_channel(row,col) = d;
            end
        end
    end
end


function cleaned_channel = clean_corrupted_pixels(channel, corrupted_pixels)
    cleaned_channel = channel;
    for row = 1:size(channel,1)
        for col = 1:size(channel,2)
            if (corrupted_pixels(row, col) == 1)
                cleaned_channel(row, col) = 0.5*(channel(row,col-1)+channel(row,col+1));
            end
        end
    end
end
