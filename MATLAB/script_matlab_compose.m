weights_data = load('weights_decomposed.mat');
weights = weights_data.weights_decomposed;
composed_weights = cell(1, 6);
for i = 1:6
    U = weights(i, :);
    T = cpdgen(U);
    composed_weights{i} = T;
save('weights_composed.mat', 'composed_weights');
end