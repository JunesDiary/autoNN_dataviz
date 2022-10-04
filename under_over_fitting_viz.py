def under_over_fitting_viz(train_scores, test_scores, values):
    # plot of train and test scores vs number of neighbors
    pyplot.plot(values, train_scores, '-o', label='Train')
    pyplot.plot(values, test_scores, '-o', label='Test')
    pyplot.legend()
    pyplot.show()
