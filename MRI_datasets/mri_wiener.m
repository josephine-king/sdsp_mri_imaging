%%
functions = common_functions;
image_idx = "1";
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
image_idx = "1";
[bad_channel1, bad_channel2, bad_channel3] = functions.get_data(image_idx, 0);
fused_bad_channels = functions.fuse_channels_simple(bad_channel1, bad_channel2, bad_channel3);

corrupted_pixels = functions.get_corrupted_pixels(fused_bad_channels, 0.9, 105);
cleaned_channels = clean_corrupted_pixels(fused_bad_channels, corrupted_pixels);

window_dims = [3,3];
[Rx, r_dx] = get_Rx_rdx(fused_bad_channels, window_dims, corrupted_pixels, 1, size(fused_bad_channels));

filtered_channels = wiener_filter(fused_bad_channels, corrupted_pixels, window_dims);
piecewise_filtered_channels = piecewise_wiener_filter(fused_bad_channels, corrupted_pixels, window_dims, 16, 4);
center_filtered_channels = center_piecewise_wiener_filter(fused_bad_channels, corrupted_pixels, window_dims, [128,64]);

%% Clean first, fuse second
% Find the corrupted lines and replace them with the average of the lines
% next to them
functions = common_functions;
image_idx = "2";
[bad_channel1, bad_channel2, bad_channel3] = functions.get_data(image_idx, 0);
fused_bad_channels = functions.fuse_channels_simple(bad_channel1, bad_channel2, bad_channel3);

window_dims = [1,3];

corrupted_pixels1 = functions.get_corrupted_pixels(bad_channel1, .9, 105);
corrupted_pixels2 = functions.get_corrupted_pixels(bad_channel2, .9, 105);
corrupted_pixels3 = functions.get_corrupted_pixels(bad_channel3, .9, 105);

cleaned_channel1 = clean_corrupted_pixels(bad_channel1, corrupted_pixels1);
cleaned_channel2 = clean_corrupted_pixels(bad_channel2, corrupted_pixels2);
cleaned_channel3 = clean_corrupted_pixels(bad_channel3, corrupted_pixels3);
cleaned_channels = functions.fuse_channels_wiener(cleaned_channel1, cleaned_channel2, cleaned_channel3);

[Rx1, r_dx1] = get_Rx_rdx(bad_channel1, window_dims, corrupted_pixels1, 1, size(fused_bad_channels));
[Rx2, r_dx2] = get_Rx_rdx(bad_channel2, window_dims, corrupted_pixels2, 1, size(fused_bad_channels));
[Rx3, r_dx3] = get_Rx_rdx(bad_channel3, window_dims, corrupted_pixels3, 1, size(fused_bad_channels));

filtered_channel1 = wiener_filter(bad_channel1, corrupted_pixels1, window_dims);
filtered_channel2 = wiener_filter(bad_channel2, corrupted_pixels2, window_dims);
filtered_channel3 = wiener_filter(bad_channel3, corrupted_pixels3, window_dims);
filtered_channels = functions.fuse_channels_wiener(filtered_channel1, filtered_channel2, filtered_channel3);

piecewise_filtered_channel1 = piecewise_wiener_filter(bad_channel1, corrupted_pixels1, window_dims, 7, 3);
piecewise_filtered_channel2 = piecewise_wiener_filter(bad_channel2, corrupted_pixels2, window_dims, 7, 3);
piecewise_filtered_channel3 = piecewise_wiener_filter(bad_channel3, corrupted_pixels3, window_dims, 7, 3);
piecewise_filtered_channels = functions.fuse_channels_wiener(piecewise_filtered_channel1, piecewise_filtered_channel2, piecewise_filtered_channel3);

center_filtered_channel1 = center_piecewise_wiener_filter(bad_channel1, corrupted_pixels1, window_dims, [256,64]);
center_filtered_channel2 = center_piecewise_wiener_filter(bad_channel2, corrupted_pixels2, window_dims, [256,64]);
center_filtered_channel3 = center_piecewise_wiener_filter(bad_channel3, corrupted_pixels3, window_dims, [256,64]);
center_filtered_channels = functions.fuse_channels_wiener(center_filtered_channel1, center_filtered_channel2, center_filtered_channel3);


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

figure(1)
axis image, 
colormap gray;
axis off
subplot(2,2,1)
imagesc(cleaned_adj_img);
title("Fused and cleaned image")
subplot(2,2,2)
imagesc(filtered_adj_img);
title("Wiener filtered image")
subplot(2,2,3)
imagesc(piecewise_filtered_adj_img);
title("Piecewise Wiener filtered image")
subplot(2,2,4)
imagesc(center_filtered_adj_img);
title("Center Wiener filtered image")

figure(2); 
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
imagesc(abs(Rx))
subplot(1,2,2)
imagesc(abs(r_dx))

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
subplot(1,3,1)
imagesc(corrupted_pixels1)
subplot(1,3,2)
imagesc(corrupted_pixels2)
subplot(1,3,3)
imagesc(corrupted_pixels3)


%% Functions

function filtered_channel = center_piecewise_wiener_filter(channel, corrupted_pixels, window_dims, center_dims)
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
    [Rx_center, r_dx_center] = get_Rx_rdx(channel, window_dims, corrupted_pixels, 1, center_dims);
    [Rx_edge, r_dx_edge] = get_Rx_rdx(channel, window_dims, corrupted_pixels, 0, center_dims);

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
    filtered_channel = channel;
    % The square window excludes the middle column, so the order is smaller
    % than the window by a factor of window_size
    row_padding = (window_dims(1)-1)/2;
    col_padding = (window_dims(2)-1)/2;
    % Find the correlation matrix and cross correlations between d and x
    [Rx, r_dx] = get_Rx_rdx(channel, window_dims, corrupted_pixels, 1, size(channel));
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

function [Rx, r_dx] = get_Rx_rdx(channel, window_dims, corrupted_pixels, center_n_edge, center_dims)
    
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
            m(:, round((window_cols + 1) / 2)) = []; 
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
    Rx = reshape(Rx, window_rows*window_cols-window_rows, window_rows*window_cols-window_rows);
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

function padchan = pad_channel(x_pad, y_pad, chan)

    y_dim = size(chan,1);
    x_dim = size(chan,2);
    x_padding = zeros(y_dim, x_pad);
    y_padding = zeros (y_pad, x_dim+x_pad*2);

    padchan = [x_padding, chan, x_padding];
    padchan = [y_padding; padchan; y_padding];
end

