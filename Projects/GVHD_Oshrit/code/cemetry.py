# def censoring_nn1(censor_df: pd.DataFrame, uncensor_df: pd.DataFrame, tag_name: str, n_epochs, k_fold: int = 0):
#     """
#
#     :param censor_df: data frame of censored samples
#     :param uncensor_df: data frame of uncensored samples
#     :param tag_name: name of tag
#     :param n_epochs: number of epochs for the train and tests
#     :return:
#
#     Run Example:
#     censoring_nn(saliva_censor_df, saliva_uncensor_df, "time_to_ttcgvhd", 100)
#     """
#     # split to train and test and load data
#     x_train_df, y_train_df, x_test_df, y_test_df, index_of_additional_parameters = split_train_test_censored_and_uncensord(
#         censor_df, uncensor_df, tag_name, additional_cols_to_keep=["ttcgvhd_for_loss"])
#
#     test_loader = DataLoader(dataset=TensorDataset(torch.from_numpy(x_test_df.values).float(),
#                                                    torch.from_numpy(y_test_df.values).float()), batch_size=20)
#     train_loader, valid_loader = split_train_validation(x_train_df, y_train_df, tag_name, k_fold=k_fold)
#     for alpha, beta in [(0, 0)]:
#         for lr in [1e-2, 1e-3, 1e-4, 1e-5]:
#             for wd in [0.1, 0.01, 0.001]:
#                 loss_fn = My_loss(alpha, beta)
#                 model = Model_luzon(len(train_loader.dataset[0][0]) + index_of_additional_parameters)
#                 optimizer = optim.Adam(model.parameters(), weight_decay=wd, lr=lr)
#                 losses = []
#                 test_losses = []
#                 valid_losses = []
#                 # Creates the train_step function for our model, loss function and optimizer
#                 train_step = make_train_step_no_last_columns(model, loss_fn, optimizer, index_of_additional_parameters)
#                 min_loss = float("inf")
#                 num_of_runs_with_no_change = NUM_OF_RUNS_WITH_NO_CHANGE_IN_LOSS
#                 for epoch in range(n_epochs):
#                     for k in range(max(k_fold, 1)):
#                         train_loader, valid_loader = split_train_validation(x_train_df, y_train_df, tag_name,
#                                                                             k_fold=k_fold)
#                         # Train
#                         total_loss = 0
#                         num_of_iters_train = 0
#                         for x_batch, y_batch in train_loader:
#                             num_of_iters_train += 1
#                             # Performs one train step and returns the corresponding loss
#                             total_loss += train_step(x_batch, y_batch)
#
#                         # Validation evaluation
#                         if k_fold > 0:
#                             with torch.no_grad():
#                                 num_of_iters_valid = 0
#                                 val_loss = 0
#                                 for x_val, y_val in valid_loader:
#                                     num_of_iters_valid += 1
#                                     model.eval()
#
#                                     yhat = model(x_val[:, :index_of_additional_parameters])
#                                     val_loss += loss_fn(y_val, yhat, x_val[:, index_of_additional_parameters:])
#                     if k_fold > 0:
#                         num_of_runs_with_no_change -= 1
#                         if min_loss > val_loss / max(1, k_fold):
#                             min_loss = val_loss
#                             num_of_runs_with_no_change = NUM_OF_RUNS_WITH_NO_CHANGE_IN_LOSS
#                         print(num_of_runs_with_no_change)
#                         if num_of_runs_with_no_change <= 0:
#                             break
#                     # End of k_fold loop
#                     # Test evaluation
#                     with torch.no_grad():
#                         num_of_iters_test = 0
#                         test_loss = 0
#                         for x_val, y_val in test_loader:
#                             num_of_iters_test += 1
#                             model.eval()
#
#                             yhat = model(x_val[:, :-1])
#                             test_loss += loss_fn(y_val, yhat, x_val[:, -1:])
#
#                     test_losses.append(test_loss / len(test_loader.dataset[0][0]))
#                     losses.append((total_loss / max(k_fold, 1)) / len(train_loader.dataset[0][0]))
#                     if k_fold > 0:
#                         valid_losses.append((val_loss / max(k_fold, 1)) / len(valid_loader.dataset[0][0]))
#
#                 # plot the loss func:
#                 matplotlib.rc('xtick', labelsize=30)
#                 matplotlib.rc('ytick', labelsize=30)
#                 plt.figure(figsize=(35, 10))
#                 plt.plot(losses, label="train")
#                 plt.plot(test_losses, label="test")
#                 if k_fold > 0:
#                     plt.plot(valid_losses, label="validation")
#                 plt.title(f"alpha: {alpha}, beta: {beta}, learning rate: {lr}, wight decay: {wd}", fontsize=30)
#                 plt.legend(fontsize=30)
#                 plt.savefig(f"plots/loss per epochs for alpha {alpha} beta {beta} lr {lr} wd {wd} line.svg")
#                 plt.show()
#
#                 matplotlib.rc('xtick', labelsize=30)
#                 matplotlib.rc('ytick', labelsize=30)
#                 plt.figure(figsize=(35, 10))
#                 plt.plot(losses, "o", label="train")
#                 plt.plot(test_losses, "o", label="test")
#                 if k_fold > 0:
#                     plt.plot(valid_losses, "o", label="validation")
#                 plt.title(f"alpha: {alpha}, beta: {beta}, learning rate: {lr}, wight decay: {wd}", fontsize=30)
#                 plt.legend(fontsize=30)
#                 plt.savefig(f"plots/loss per epochs for alpha {alpha} beta {beta} lr {lr} wd {wd} scatter.svg")
#                 plt.show()
#
#                 print(losses[-1:])
#                 print(test_losses[-1:])