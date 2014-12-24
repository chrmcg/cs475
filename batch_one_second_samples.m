% Shorten all audio files in specified folder to the first one second of
% audio and output new audio file to specified folder at 24-bits per sample

clc; clear all; close all;
files = dir('/Users/Neil/School/College/Junior S1/Machine Learning/Final/raw_data');

for i = 5:length(files)
    file = files(i);
    filename = file.name;
    
    audio = audioread(strcat('raw_data/', filename));
    length = size(audio);
    if length(1) < 44100
        first_second = audio;
        first_second(length(1)+1:44100,:) = 0;
    else
        first_second = audio(1:44100,:);
    end
    
    audiowrite(strcat('raw_data_2/', filename), first_second, 44100, 'BitsPerSample', 24);
end