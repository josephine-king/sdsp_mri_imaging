clc; clear all; close all;

load('MRI_datasets/Slice4/GoodData/slice4_channel1.mat');
load('MRI_datasets/Slice4/GoodData/slice4_channel2.mat');
load('MRI_datasets/Slice4/GoodData/slice4_channel3.mat');
load('MRI_datasets/Slice4/BadData/slice4_channel1.mat');
load('MRI_datasets/Slice4/BadData/slice4_channel2.mat');
load('MRI_datasets/Slice4/BadData/slice4_channel3.mat');

 

 C= spiral_matrix(200,100,10,200, 5);
close all
figure(1); 
imagesc(C);


%%

n = 512*128; % dimension of s
p = round(n/100);
kspace = reshape(slice4_channel1_goodData, n, 1);
%C = sparse_matrix(p,n,p*500);
%C = center_matrix(p,n);
C = radial_matrix(p,n,30,50);
%C = spiral_matrix(p,n,50,100000,15);
y = C*kspace;

close all
axis image, 
figure(1); 
imagesc(C);

%%
% L1 minimum norm solution s_L1
C = double(C);
y = double(y);
cvx_begin;
    variable s_L1(n) complex;
    minimize(norm(s_L1,1));
    subject to 
        norm(C*s_L1 - y, 2) < 10;
cvx_end;

%s_L2 = pinv(Theta)*y; % L2 minimum norm solution

% Example of Steve Brunton code
% https://stackoverflow.com/questions/66510827/compressed-sensing-why-does-my-convex-solver-fail-when-i-give-it-more-sample

%%
% IFFT of k-space data
%channel 1 (replace "slice1_channel1_goodData" with
%slice1_channel1_badData) for bad images
s_L1 = reshape(s_L1, 512, 128)
Data_img(:,:,1) = ifftshift(ifft2(s_L1),1);

% clear compensation, preparation, based on fourier transformed blinked 
% k-space data (Data_raw)
clear_comp = linspace(10,0.1,size(Data_img,2)).^2; 
clear_matrix = repmat(clear_comp,[size(Data_img,1) 1]);

% combine 3 channels sum of squares and add clear compensation
eye_raw  = sqrt( abs(squeeze(Data_img(:,:,1))).^2).* clear_matrix;  
    
% crop images because we are only interested in eye. Make it square 
% 128 x 128
crop_x = [128 + 60 : 348 - 33]; % crop coordinates
eye_raw = eye_raw(crop_x, :);

% Visualize the images. 

%image
eye_visualize = reshape(squeeze(eye_raw(:,:)),[128 128]); 


% For better visualization and contrast of the eye images, histogram based
% compensation will be done 

std_within = 0.995; 
% set maximum intensity to contain 99.5 % of intensity values per image
[aa, val] = hist(eye_visualize(:),linspace(0,max(...
                                    eye_visualize(:)),1000));
    thresh = val(find(cumsum(aa)/sum(aa) > std_within,1,'first'));
    
% set threshold value to 65536
eye_visualize = uint16(eye_visualize * 65536 / thresh); 

close all
figure(1); 
imagesc(eye_visualize(:,:,1));
axis image, 
colormap gray;
axis off

figure(2); 
imagesc(100*log(abs(s_L1)));

figure(3); 
imagesc(C);

%%
% Calculate MSE


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
