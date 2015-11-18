function [ predicted_label ] = logistic( train_x, train_y, test_x)
    model = train(train_y, sparse(train_x), ['-s 0', 'col']);
    [predicted_label] = predict(round(rand(size(test_x,1),1)), sparse(test_x), model, ['-q', 'col']);
end

