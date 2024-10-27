functions = common_functions;
image_idx = "1";
[good_channel1, good_channel2, good_channel3] = functions.get_data(image_idx, 1);

fused_channels = functions.fuse_channels_wiener(good_channel1, good_channel2, good_channel3);

% Sparsity factor - how much we are reducting the original measurement size
f = 10;
n = size(fused_channels,1)*size(fused_channels,2);
error = 50;

% Solve convex optimization problem for compressed sensing to get the
% sparse vector s
%%
[C_radial, s_radial] = compressed_sensing(fused_channels, f, "radial", [30,50], error);
%%
[C_center, s_center] = compressed_sensing(fused_channels, f, "center", [], error);
%%
[C_spiral, s_spiral] = compressed_sensing(fused_channels, f, "spiral", [300,100000,25], error);
%%
[C_sparse, s_sparse] = compressed_sensing(fused_channels, f, "sparse", [500], error);


%% Get smaller C matrices for plotting
C_radial_plot = radial_matrix(6552/4,65536/8,30,10);
C_center_plot = center_matrix(6552/4,65536/8);
C_spiral_plot = spiral_matrix(6552/4,65536/8, 300, 100000, 25);
C_sparse_plot = sparse_matrix(6552/4,65536/8, 6552/4*500);

%%
good_img = functions.get_image(fused_channels);
good_adj_img = functions.adjust_image(good_img, 1);
[radial_img, radial_MSE] = functions.get_image_and_mse(s_radial, good_adj_img);
[center_img, center_MSE] = functions.get_image_and_mse(s_center, good_adj_img);
[spiral_img, spiral_MSE] = functions.get_image_and_mse(s_spiral, good_adj_img);
[sparse_img, sparse_MSE] = functions.get_image_and_mse(s_sparse, good_adj_img);

%% Plotting

figure (1)
subplot(1,4,1)
imagesc(100*log(abs(s_radial)));
title("K-space resulting from radial PSF")
subplot(1,4,2)
imagesc(100*log(abs(s_center)));
title("K-space resulting from center PSF")
subplot(1,4,3)
imagesc(100*log(abs(s_spiral)));
title("K-space resulting from spiral PSF")
subplot(1,4,4)
imagesc(100*log(abs(s_sparse)));
title("K-space resulting from sparse PSF")
%%
figure (2)
axis image, 
colormap gray;
axis off
subplot(1,4,1)
imagesc(radial_img);
title("Image resulting from radial PSF")
subplot(1,4,2)
imagesc(center_img);
title("Image resulting from center PSF")
subplot(1,4,3)
imagesc(spiral_img);
title("Image resulting from spiral PSF")
subplot(1,4,4)
imagesc(sparse_img);
title("Image resulting from sparse PSF")
%%
figure (3)
subplot(1,4,1)
imagesc(C_radial_plot);
title("Radial PSF")
subplot(1,4,2)
imagesc(C_center_plot);
title("Center PSF")
subplot(1,4,3)
imagesc(C_spiral_plot);
title("Spiral PSF")
subplot(1,4,4)
imagesc(C_sparse_plot);
title("Sparse PSF")

%% Functions

function [C, s] = compressed_sensing(chan, n_over_p, psf, psf_settings, error)
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
    C = double(C);
    y = double(y);
    cvx_begin;
        variable s(n) complex;
        minimize(norm(s,1));
        subject to 
            norm(C*s - y, 2) <= error;
    cvx_end;
    
    s = reshape(s, size(chan,1), size(chan,2));
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
            rand_num = rand/32;

            if (normalized_distance*rand < rand_num)
                C(j, i) = 1;
            end
        end 
    end
 end

 function C = radial_matrix(rows, cols, n, e)
    num_cells = rows*cols;
    C = zeros(rows, cols);
   
    % use row = m*col to create radial lines coming from the center
    m = linspace(-1,1,n);
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

