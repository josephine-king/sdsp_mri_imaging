% Fuses and filters using a Wiener filter
% channels1-3 are the k-space channels, and cp1-3 are their corresponding
% corrupted pixel matrices.
% fuse_first - if 1, fuse before filtering. otherwsie, filter before fusing
% fuse_type - can be noise, average, snr, or wiener
% window_dims - size of the window sliding over the k-space
% wiener_type - can be wiener, piecewise, or center_piecewise
% wiener_type_args - some additional arguments that are needed for the
% piecewise and center_piecewise filters
function filtered_channels = mri_wiener(channel1, channel2, channel3, cp1, cp2, cp3, fuse_first, fuse_type, window_dims, wiener_type, wiener_type_args)
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

% Applies a Wiener filter to a k-space channel by sliding a window with
% window_dims over the space
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

% Applies a center piecewise Wiener filter to a k-space channel by sliding a window with
% window_dims over the space. center_dims specifies how large the center
% rectangle is 
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

% Applies a piecewise Wiener filter to a k-space channel by sliding a window with
% window_dims over the space. num_rows and num_cols specify how the k-space
% should be divided for the piecewise statistical calculations. 
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