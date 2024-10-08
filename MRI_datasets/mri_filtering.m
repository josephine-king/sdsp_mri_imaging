%% Script to translate the K-spcae images into spatial eye images
clc; clear all; close all;

load('/Users/josephineking/Desktop/sdsp_mri_imaging/MRI_datasets/Slice1/GoodData/slice1_channel1.mat');
load('/Users/josephineking/Desktop/sdsp_mri_imaging/MRI_datasets/Slice1/GoodData/slice1_channel2.mat');
load('/Users/josephineking/Desktop/sdsp_mri_imaging/MRI_datasets/Slice1/GoodData/slice1_channel3.mat');
load('/Users/josephineking/Desktop/sdsp_mri_imaging/MRI_datasets/Slice1/BadData/slice1_channel1.mat');
load('/Users/josephineking/Desktop/sdsp_mri_imaging/MRI_datasets/Slice1/BadData/slice1_channel2.mat');
load('/Users/josephineking/Desktop/sdsp_mri_imaging/MRI_datasets/Slice1/BadData/slice1_channel3.mat');


% 1. X - dimension of the K-Space data    - 128
% 2. Y - dimension of the K-Space data    - 512

% IFFT of k-space data
%channel 1 (replace "slice1_channel1_goodData" with
%slice1_channel1_badData) for bad images
%smoothed1 = smoothdata2(slice4_channel1_badData, "lowess", 3);
%smoothed2 = smoothdata2(slice4_channel2_badData, "lowess", 3);
%smoothed3 = smoothdata2(slice4_channel3_badData, "lowess", 3);
x_pad = zeros(512, 5);
y_pad = zeros (20, 128+10);
smoothed1 = slice1_channel1_goodData;
smoothed2 = slice1_channel2_goodData;
smoothed3 = slice1_channel3_goodData;
smoothed1 = [x_pad, smoothed1, x_pad];
smoothed1 = [y_pad; smoothed1; y_pad];
smoothed2 = [x_pad, smoothed2, x_pad];
smoothed2 = [y_pad; smoothed2; y_pad];
smoothed3 = [x_pad, smoothed3, x_pad];
smoothed3 = [y_pad; smoothed3; y_pad];
smoothed1 = smoothed1.*(hamming(552, "periodic")*hamming(138, "periodic")');
smoothed2 = smoothed2.*(hamming(552, "periodic")*hamming(138, "periodic")');
smoothed3 = smoothed3.*(hamming(552, "periodic")*hamming(138, "periodic")');
%smoothed1 = sgolayfilt(slice4_channel1_badData, 20, 11);
%smoothed2 = sgolayfilt(slice4_channel2_badData, 20, 11);
%smoothed3 = sgolayfilt(slice4_channel3_badData, 20, 11);

Data_img(:,:,1) = ifftshift(ifft2(smoothed1),1);
%channel 2
Data_img(:,:,2) = ifftshift(ifft2(smoothed2),1);
%channel 3
Data_img(:,:,3) = ifftshift(ifft2(smoothed3),1);

% clear compensation, preparation, based on fourier transformed blinked 
% k-space data (Data_raw)
clear_comp = linspace(10,0.1,size(Data_img,2)).^2; 
clear_matrix = repmat(clear_comp,[size(Data_img,1) 1]);

% combine 3 channels sum of squares and add clear compensation
eye_raw  = sqrt( abs(squeeze(Data_img(:,:,1))).^2 + ...
           abs(squeeze(Data_img(:,:,2))).^2 + ...
           abs(squeeze(Data_img(:,:,3))).^2).* clear_matrix;  
    
% crop images because we are only interested in eye. Make it square 
% 128 x 128
crop_x = [128 + 60 : 348 - 33 + 10]; % crop coordinates
eye_raw = eye_raw(crop_x, :);

% Visualize the images. 

%image
eye_visualize = reshape(squeeze(eye_raw(:,:)),[138 138]); 


% For better visualization and contrast of the eye images, histogram based
% compensation will be done 

std_within = 0.995; 
% set maximum intensity to contain 99.5 % of intensity values per image
[aa, val] = hist(eye_visualize(:),linspace(0,max(...
                                    eye_visualize(:)),1000));
    thresh = val(find(cumsum(aa)/sum(aa) > std_within,1,'first'));
    
% set threshold value to 65536
eye_visualize = uint16(eye_visualize * 65536 / thresh); 


%% plotting scripts
close all
figure(1); 
imagesc(eye_visualize(:,:,1));
axis image, 
colormap gray;
axis off

% Spatial frequency observations
figure(2); 
imagesc(100*log(abs(smoothed1)));

figure(3); 
imagesc((hamming(512, "periodic")*hamming(128, "periodic")'));

xlabel('Horizontal frequency bins')
ylabel('Vertical frequency bins');

