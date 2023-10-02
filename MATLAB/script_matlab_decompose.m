weights_data = load('weights_base.mat');
weights = weights_data.weights_base;
options = struct;
options.Algorithm = @cpd_nls;
weights_nls = cell(1, 15);
for i = 1:length(weights)
    array = weights{1, i};
    sizes = size(array);
    R = fix(sizes(1) / 3);
    U = cpd_rnd(sizes, R);
    U_nls = cpd(array, U, options);
    weights_nls{i} = U_nls;
save('weights_nls.mat', 'weights_nls');
end