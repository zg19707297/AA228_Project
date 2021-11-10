clear all;
close all;
clc;

x = [1:20];
y = [1:20];
U = readtable("ValueIteration.csv");
h = heatmap(x,y,U{:,:},'Colormap',jet);