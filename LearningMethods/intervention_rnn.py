import os
from collections import Counter
from random import shuffle
from sys import stdout
import pandas as pd
import numpy as np
import nni
import torch
from matplotlib.pyplot import figure
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.nn import Module, LSTM, Linear
from torch.utils.data import Dataset
from torch.nn.functional import mse_loss, relu, sigmoid, leaky_relu
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from LearningMethods.model_params import MLPParams, RNNActivatorParams, RNNModuleParams, MicrobiomeDataset, \
    split_microbiome_dataset
import  math
from  sklearn import utils
TRAIN_JOB = "TRAIN"
DEV_JOB = "DEV"
VALIDATE_JOB = "VALIDATE"
LOSS_PLOT = "loss"
CORR_PLOT = "correlation"
ACCURACY_PLOT = "accuracy"
R2_PLOT = "R^2"
CALC_CORR = True
CALC_R2 = False
CALC_ACC = False
AUC_PLOT = "ROC-AUC"
CONV_LAYER = False
SHUFFLE = True
DIM = 2

NUMBER_OF_BACTERIA = 0
NUMBER_OF_TIME_POINTS = 0
NUMBER_OF_SAMPLES = 0

PRINT_PROGRESS = False
PRINT_INFO = False



# ----------------------------------------------- models -----------------------------------------------
class MicrobiomeModule(Module):
    def __init__(self, params: RNNModuleParams):
        super(MicrobiomeModule, self).__init__()
        # useful info in forward function
        self.params = params
        self.task = params.TASK
        self._sequence_lstm = SequenceModule(params.SEQUENCE_PARAMS)
        if self.task == 'reg':
            self._mlp = MLPModule(params.LINEAR_PARAMS)
        elif self.task == 'class':
            self._mlp = MLPModuleClass(params.LINEAR_PARAMS)
        self.optimizer = self.set_optimizer(params.LEARNING_RATE, params.OPTIMIZER, params.REGULARIZATION)

    def set_optimizer(self, lr, opt, l2_reg):
        return opt(self.parameters(), lr=lr, weight_decay=l2_reg)

    def forward(self, x):
        x = self._sequence_lstm(x)
        x = self._mlp(x)
        """
        print("lstm dim:")
        print(x.shape)
        print("mlp dim:")
        print(x.shape)
        """
        return x


class SequenceModule(Module):
    def __init__(self, params):
        super(SequenceModule, self).__init__()
        self._lstm = LSTM(input_size=params.LSTM_input_dim, hidden_size=params.LSTM_hidden_dim, num_layers=params.LSTM_layers, batch_first=True,
                          bidirectional=True, dropout=params.LSTM_dropout)

    def forward(self, x):
        # 3 layers LSTM
        """
        print("lstm:")
        print(self._lstm)
        """
        output_seq, _ = self._lstm(x.float())
        """
        print("output seq:")
        print(output_seq.shape)
        """
        return output_seq   # for single bacteria model .transpose(1, 2)


class MLPModule(Module):
    def __init__(self, params: MLPParams):
        super(MLPModule, self).__init__()
        # useful info in forward function
        self._linear = Linear(params.LINEAR_in_dim * 2, params.LINEAR_out_dim)

    def forward(self, x):
        x = self._linear(x)
        return x
class MLPModuleClass(Module):
    def __init__(self, params: MLPParams):
        super(MLPModule, self).__init__()
        # useful info in forward function
        self._linear = Linear(params.LINEAR_in_dim * 2, params.LINEAR_out_dim)

    def forward(self, x):
        x = self._linear(x)
        return torch.sigmoid(x)


# ----------------------------------------------- activate model -----------------------------------------------
class Activator:
    def __init__(self, model: MicrobiomeModule, params: RNNActivatorParams, data: MicrobiomeDataset, splitter):
        self._model = model.cuda() if params.GPU else model
        self._gpu = params.GPU
        self._epochs = int(params.EPOCHS)
        self._batch_size = params.BATCH_SIZE
        self._loss_func = params.LOSS
        self._corr_func = params.CORR
        self._r2_func = params.R2
        self._early_stop = params.EARLY_STOP
        self.shuffle = model.params.SHUFFLE
        # self._load_data(data, params.TRAIN_TEST_SPLIT, params.BATCH_SIZE, splitter)
        self._init_loss_and_acc_vec()
        self._init_print_att()
        self._dim = model.params.DIM

    # init loss and accuracy vectors (as function of epochs)
    def _init_loss_and_acc_vec(self):
        self._loss_vec_train = []
        self._loss_vec_dev = []
        self._corr_vec_train = []
        self._corr_vec_dev = []
        self._r2_vec_train = []
        self._r2_vec_dev = []
        self._accuracy_vec_train = []
        self._accuracy_vec_dev = []
        self._auc_vec_train = []
        self._auc_vec_dev = []

    # init variables that holds the last update for loss and accuracy
    def _init_print_att(self):
        self._print_train_loss = 0
        self._print_dev_loss = 0
        self._print_train_corr = 0
        self._print_dev_corr = 0
        self._print_train_r2 = 0
        self._print_dev_r2 = 0
        self._print_train_accuracy = 0
        self._print_train_auc = 0
        self._print_dev_accuracy = 0
        self._print_dev_auc = 0

    # update loss after validating
    def _update_loss(self, loss, job=TRAIN_JOB):
        if job == TRAIN_JOB:
            self._loss_vec_train.append(loss)
            self._print_train_loss = loss
        elif job == DEV_JOB:
            self._loss_vec_dev.append(loss)
            self._print_dev_loss = loss

    # update correlation after validating
    def _update_corr(self, corr, job=TRAIN_JOB):
        if job == TRAIN_JOB:
            self._corr_vec_train.append(corr)
            self._print_train_corr = corr
        elif job == DEV_JOB:
            self._corr_vec_dev.append(corr)
            self._print_dev_corr = corr

    # update r2 after validating
    def _update_r2(self, r2, job=TRAIN_JOB):
        if job == TRAIN_JOB:
            self._r2_vec_train.append(r2)
            self._print_train_r2 = r2
        elif job == DEV_JOB:
            self._r2_vec_dev.append(r2)
            self._print_dev_r2 = r2

    # update accuracy after validating
    def _update_accuracy(self, pred, true, job=TRAIN_JOB):
        # calculate acc
        acc = sum([1 if round(i) == int(j) else 0 for i, j in zip(pred, true)]) / len(pred)
        if job == TRAIN_JOB:
            self._print_train_accuracy = acc
            self._accuracy_vec_train.append(acc)
            return acc
        elif job == DEV_JOB:
            self._print_dev_accuracy = acc
            self._accuracy_vec_dev.append(acc)
            return acc

    # update auc after validating
    def _update_auc(self, pred, true, job=TRAIN_JOB):
        # pred_ = [-1 if np.isnan(x) else x for x in pred]
        # if there is only one class in the batch
        num_classes = len(Counter(true))
        if num_classes < 2:
            auc = 0.5
        # calculate auc
        else:
            auc = roc_auc_score(true, pred)
        if job == TRAIN_JOB:
            self._print_train_auc = auc
            self._auc_vec_train.append(auc)
            return auc
        elif job == DEV_JOB:
            self._print_dev_auc = auc
            self._auc_vec_dev.append(auc)
            return auc

    # print progress of a single epoch as a percentage
    def _print_progress(self, batch_index, len_data, job=""):
        if PRINT_PROGRESS:
            prog = int(100 * (batch_index + 1) / len_data)
            stdout.write("\r\r\r\r\r\r\r\r" + job + " %d" % prog + "%")
            print("", end="\n" if prog == 100 else "")
            stdout.flush()

    # print last loss and accuracy
    def _print_info(self, jobs=()):
        if PRINT_INFO:
            if TRAIN_JOB in jobs:
                print("Loss_Train: " + '{:{width}.{prec}f}'.format(self._print_train_loss, width=6, prec=4),
                      end=" || ")
                if CALC_CORR:
                    print("Corr_Train: " + '{:{width}.{prec}f}'.format(self._print_train_corr, width=6, prec=4),
                      end=" || ")
                if CALC_R2:
                    print("R2_Train: " + '{:{width}.{prec}f}'.format(self._print_train_r2, width=6, prec=4),
                      end=" || ")
                if CALC_ACC:
                    print("Acc_Train: " + '{:{width}.{prec}f}'.format(self._print_train_accuracy, width=6, prec=4) +
                      " || AUC_Train: " + '{:{width}.{prec}f}'.format(self._print_train_auc, width=6, prec=4) +
                      " || ")

            if DEV_JOB in jobs:
                print("Loss_Dev: " + '{:{width}.{prec}f}'.format(self._print_dev_loss, width=6, prec=4),
                      end=" || ")
                if CALC_CORR:
                    print("Corr_Dev: " + '{:{width}.{prec}f}'.format(self._print_dev_corr, width=6, prec=4),
                      end=" || ")
                if CALC_R2:
                    print("R2_Dev: " + '{:{width}.{prec}f}'.format(self._print_dev_r2, width=6, prec=4),
                      end=" || ")
                if CALC_ACC:
                    print("Acc_Dev: " + '{:{width}.{prec}f}'.format(self._print_dev_accuracy, width=6, prec=4) +
                      " || AUC_Dev: " + '{:{width}.{prec}f}'.format(self._print_dev_auc, width=6, prec=4) +
                      " || ")
            print("")

    # plot loss / accuracy graph
    def plot_line(self, title, save_name, job=LOSS_PLOT):

        if job == LOSS_PLOT:
            y_axis_train = self._loss_vec_train  # if job == LOSS_PLOT else self._accuracy_vec_train
            y_axis_dev = self._loss_vec_dev  # if job == LOSS_PLOT else self._accuracy_vec_dev
            c1 = "red"
            c2 = "blue"
        elif job == CORR_PLOT:
            y_axis_train = self._corr_vec_train
            y_axis_dev = self._corr_vec_dev
            c1 = "c"
            c2 = "m"
        elif job == R2_PLOT:
            y_axis_train = self._r2_vec_train
            y_axis_dev = self._r2_vec_dev
            c1 = "green"
            c2 = "orange"

        elif job == ACCURACY_PLOT:
            y_axis_train = self._accuracy_vec_train
            y_axis_dev = self._accuracy_vec_dev
            c1 = "brown"
            c2 = "pink"

        elif job == AUC_PLOT:
            y_axis_train = self._auc_vec_train
            y_axis_dev = self._auc_vec_dev
            c1 = "gray"
            c2 = "olive"

        fig, ax = plt.subplots()
        plt.title(title)
        y = np.linspace(0, len(y_axis_train), len(y_axis_train))
        plt.plot(y, y_axis_train, label="train", color=c1)
        plt.plot(y, y_axis_dev, label="test", color=c2)
        plt.xlabel("epochs")
        plt.xticks(np.arange(50, len(y_axis_train) + 1, step=50))
        plt.ylabel(job)
        plt.legend()
        plt.savefig(save_name + " " + job + ".png")
        # plt.show()
    def _plot_acc_dev(self):
        self.plot_line(title="loss", save_name="plot", job=LOSS_PLOT)
        self.plot_line(title="R2", save_name="plot", job=R2_PLOT)
        self.plot_line(title="accurecy", save_name="plot", job=ACCURACY_PLOT)
        self.plot_line(title="auc", save_name="plot", job=AUC_PLOT)

    @property
    def model(self):
        return self._model

    @property
    def loss_train_vec(self):
        return self._loss_vec_train

    @property
    def loss_dev_vec(self):
        return self._loss_vec_dev

    @property
    def corr_train_vec(self):
        return self._corr_vec_train

    @property
    def corr_dev_vec(self):
        return self._corr_vec_dev

    @property
    def r2_train_vec(self):
        return self._r2_vec_train

    @property
    def r2_dev_vec(self):
        return self._r2_vec_dev

    @property
    def accuracy_train_vec(self):
        return self._accuracy_vec_train

    @property
    def auc_train_vec(self):
        return self._auc_vec_train


    @property
    def accuracy_dev_vec(self):
        return self._accuracy_vec_dev

    @property
    def auc_dev_vec(self):
        return self._auc_vec_dev


    # load dataset
    def _load_data(self, train_dataset, train_split, batch_size, splitter):
        # split dataset
        train, dev = splitter(train_dataset, [train_split, 1-train_split])
        # set train loader
        self._train_loader = DataLoader(
            train,
            batch_size=batch_size,
            # collate_fn=train.collate_fn,
            shuffle=SHUFFLE,
        )

        self._train_valid_loader = DataLoader(
            train,
            batch_size=1000,
            # collate_fn=train.collate_fn,
            shuffle=SHUFFLE,
        )

        # set validation loader
        self._dev_loader = DataLoader(
            dev,
            batch_size=1000,
            # collate_fn=dev.collate_fn,
            shuffle=SHUFFLE,
        )

    def _to_gpu(self, x, l):
        x = x.cuda() if self._gpu else x
        l = l.cuda() if self._gpu else l
        return x, l

    # train a model, input is the enum of the model type
    def train(self, x_train, y_train, x_test, y_test, show_plot=False, apply_nni=False, validate_rate=10):
        self._init_loss_and_acc_vec()
        # calc number of iteration in current epoch
        # len_data = len(self._train_loader)
        last_epoch = list(range(self._epochs))[-1]
        for epoch_num in range(self._epochs):
            print(epoch_num)
            # calc number of iteration in current epoch
            x_train, y_train = utils.shuffle(x_train, y_train, random_state=0)
            for (sequence, label) in zip(x_train, y_train):
                sequence, label = self._to_gpu(sequence, label)
                # print progress
                self._model.train()
                output = self._model(sequence)                  # calc output of current model on the current batch

                loss = self._loss_func(output.squeeze(dim=self._dim), label.float())  # calculate loss
                # print(loss)
                loss.backward()                                 # back propagation
                self._model.optimizer.step()                    # update weights
                self._model.optimizer.zero_grad()                         # zero gradients

                self._train_label_and_output = (label, output)
            # validate and print progress

            # /----------------------  FOR NNI  -------------------------
            if epoch_num % validate_rate == 0:
                # validate on dev set anyway
                save_true_and_pred = True
                self._validate(x_test, y_test, save_true_and_pred, job=DEV_JOB)
                torch.cuda.empty_cache()
                # report dev result as am intermediate result
                if apply_nni:
                    test_loss = self._print_dev_loss
                    nni.report_intermediate_result(test_loss)
                # validate on train set as well and display results
                else:
                    torch.cuda.empty_cache()
                    self._validate(x_train, y_train, save_true_and_pred, job=TRAIN_JOB)
                    self._print_info(jobs=[TRAIN_JOB, DEV_JOB])

            if self._early_stop and epoch_num > 30 and self._print_dev_loss > np.max(self._loss_vec_dev[-10:]):
                break

        # report final results
        if apply_nni:
            test_loss = np.max(self._print_dev_loss)
            nni.report_final_result(test_loss)

        if show_plot:
            self._plot_acc_dev()
    # validation function only the model and the data are important for input, the others are just for print
    def _validate(self,x_test, y_test, save_true_and_pred, job=""):
        # for calculating total loss and accuracy
        loss_count = 0
        corr_count = 0
        r2_count = 0
        true_labels = []
        pred = []

        self._model.eval()
        # calc number of iteration in current epoch
        # len_data = len(data_loader)
        len_data = 0
        len_data_corr = 0
        for (full_sequence, full_label) in zip(x_test, y_test):
            for (sequence, label) in zip(full_sequence, full_label):
                sequence = torch.unsqueeze(sequence,0)
                len_data += 1
                sequence, label = self._to_gpu(sequence, label)

                output = self._model(sequence)
                # calculate total loss
                loss_count += self._loss_func(output.squeeze(dim=2), label.float())   # calculate loss
                # if CALC_CORR:
                #     c = self._corr_func(output.squeeze(dim=2).float(), label.float()) * len(label) if not math.isnan(self._corr_func(output.squeeze(dim=2).float(), label.float())) else 0
                #     if c != 0:
                #         len_data_corr += len(label)
                #         corr_count += c
                # if CALC_R2:
                #     r2_count += self._r2_func(output.squeeze(dim=2).float(), label.float()) * len(label)  # calculate r^2

                true_labels.append(label.tolist())
                pred.append(output.squeeze().tolist())

                if save_true_and_pred:
                    self._test_label_and_output = (label, output)

        # update loss accuracy
        loss = float(loss_count / len_data)
        self._update_loss(loss, job=job)

        if CALC_CORR:
            corr = self._corr_func(pred, true_labels)
            self._update_corr(corr, job=job)

        if CALC_R2:
            r2 = float(r2_count / len_data)
            self._update_r2(r2, job=job)

        """
        self._update_accuracy(pred, true_labels, job=job)
        self._update_auc(pred, true_labels, job=job)
        """
        return loss

    def scatter_results(self, fig_title, folder_and_name):
        if self._train_label_and_output and self._test_label_and_output:
            train_l, train_o = self._train_label_and_output
            test_l, test_o = self._test_label_and_output
            train_l, train_o = train_l.flatten().detach().numpy(), train_o.flatten().detach().numpy()
            test_l, test_o = test_l.flatten().detach().numpy(), test_o.flatten().detach().numpy()
            min_val = min(min(np.min(train_l), np.min(train_o)), min(np.min(test_l), np.min(test_o)))
            max_val = max(max(np.max(train_l), np.max(train_o)), max(np.max(test_l), np.max(test_o)))
            fig, ax = plt.subplots()
            plt.ylim((min_val, max_val))
            plt.xlim((min_val, max_val))
            plt.scatter(train_l, train_o,
                        label='Train', color='red', s=3, alpha=0.3)
            plt.scatter(test_l, test_o,
                        label='Test', color='blue', s=3, alpha=0.3)
            plt.title(fig_title, fontsize=10)
            plt.xlabel('Real values')
            plt.ylabel('Predicted Values')
            plt.legend(loc='upper left')
            print(folder_and_name)
            plt.savefig(folder_and_name)
            # plt.show()

    def scatter_results_by_time_point(self, fig_title, folder_and_name):
        """
        a = np.array(
            [[[1, 2, 3, 4], [5, 6, 7, 8]],
             [[9, 10, 11, 12], [13, 14, 15, 16]],
             [[17, 18, 19, 20], [21, 22, 23, 24]]])
        t = np.split(a, indices_or_sections=2, axis=1)
        b = np.split(a, indices_or_sections=4, axis=2)
        """
        if self._train_label_and_output and self._test_label_and_output:
            train_l, train_o = self._train_label_and_output
            test_l, test_o = self._test_label_and_output

            train_l, train_o = train_l.detach().numpy(), train_o.detach().numpy()
            test_l, test_o = test_l.detach().numpy(), test_o.detach().numpy()
            wanted_indices = train_l.shape[1]
            wanted_axis = 1
            train_l_by_time_point = np.split(train_l, indices_or_sections=wanted_indices, axis=wanted_axis)
            train_o_by_time_point = np.split(train_o, indices_or_sections=wanted_indices, axis=wanted_axis)
            test_l_by_time_point = np.split(test_l, indices_or_sections=wanted_indices, axis=wanted_axis)
            test_o_by_time_point = np.split(test_o, indices_or_sections=wanted_indices, axis=wanted_axis)

            # train_l_by_bacteria = np.split(train_l, indices_or_sections=train_l.shape[2], axis=2)

            for i, (train_l, train_o, test_l, test_o) in enumerate(zip(train_l_by_time_point, train_o_by_time_point,
                                              test_l_by_time_point, test_o_by_time_point)):
                train_l, train_o = train_l.flatten(), train_o.flatten()
                test_l, test_o = test_l.flatten(), test_o.flatten()

                min_val = min(min(np.min(train_l), np.min(train_o)), min(np.min(test_l), np.min(test_o)))
                max_val = max(max(np.max(train_l), np.max(train_o)), max(np.max(test_l), np.max(test_o)))
                fig, ax = plt.subplots()
                plt.ylim((min_val, max_val))
                plt.xlim((min_val, max_val))
                plt.scatter(train_l, train_o,
                            label='Train', color='red', s=3, alpha=0.3)
                plt.scatter(test_l, test_o,
                            label='Test', color='blue', s=3, alpha=0.3)
                plt.title(fig_title + " - Time Point " + str(i+1), fontsize=10)
                plt.xlabel('Real values')
                plt.ylabel('Predicted Values')
                plt.legend(loc='upper left')
                print(folder_and_name)
                plt.savefig(folder_and_name + "_time_Point_" + str(i+1) + ".png")
                # plt.show()

# ----------------------------------------------- run model -----------------------------------------------
def run_rnn_experiment(X, Y, missing_values, params, folder, GPU_flag, task_id,unique):
    x_train = X[0]
    x_test = X[1]
    y_train = Y[0]
    y_test = Y[1]
    NUMBER_OF_BACTERIA = x_train[0].shape[-1]
    if task_id == 'reg':
        out_dim = 1 if len(y_train[0].shape) == 2 else  y_train[0].shape[-1]
    if task_id == 'class':
        out_dim = unique
    dim = 2
    if task_id == 'reg':
        CORR_FUNC = "single_bacteria_lstm_corr" if len(y_train[0].shape) == 2 else "multi_bacteria_lstm_corr"
    if task_id == 'class':
        CORR_FUNC = 'top_1_accuracy'
    EARLY_STOP = 1
    if task_id == 'reg':
        LOSS = "custom_rmse_for_missing_values"
    if task_id == 'class':
        LOSS = "CE"
    BATCH_SIZE = 16
    structure = params["STRUCTURE"]
    layer_num = int(structure[0:3])
    hid_dim = int(structure[4:7])
    params_str = params.__str__().replace(" ", "").replace("'", "") + "_" + task_id

    # microbiome_dataset = MicrobiomeDataset(X, Y)
    activator_params = RNNActivatorParams(TRAIN_TEST_SPLIT=params["TRAIN_TEST_SPLIT"],
                                          LOSS=LOSS,
                                          CORR=CORR_FUNC,
                                          BATCH_SIZE=BATCH_SIZE,
                                          GPU=GPU_flag,
                                          EPOCHS=params["EPOCHS"],
                                          EARLY_STOP=EARLY_STOP)
    activator = Activator(MicrobiomeModule(RNNModuleParams(NUMBER_OF_BACTERIA=NUMBER_OF_BACTERIA,
                                                           lstm_hidden_dim=hid_dim,
                                                           mlp_out_dim=out_dim,
                                                           LSTM_LAYER_NUM=layer_num,
                                                           DROPOUT=params["DROPOUT"],
                                                           LEARNING_RATE=params["LEARNING_RATE"],
                                                           OPTIMIZER=params["OPTIMIZER"],
                                                           REGULARIZATION=params["REGULARIZATION"],
                                                           DIM=dim, SHUFFLE=SHUFFLE, TASK=task_id)),
                          activator_params, None, split_microbiome_dataset)

    activator.train(x_train, y_train, x_test, y_test,validate_rate=1)

    print(params_str)
    results_sub_folder = os.path.join(folder, "RNN_RESULTS")
    if not os.path.exists(results_sub_folder):
        os.makedirs(results_sub_folder)

    with open(os.path.join(results_sub_folder, params_str + "_score.txt"), "w") as s:
        s.write("loss," + ",".join([str(v) for v in range(len(activator.loss_train_vec))]) + "\n")
        s.write("train_loss," + ",".join([str(v) for v in activator.loss_train_vec]) + "\n")
        s.write("dev_loss," + ",".join([str(v) for v in activator.loss_dev_vec]) + "\n")

    df = pd.read_csv(os.path.join(results_sub_folder, params_str + "_score.txt"))
    df.to_csv(os.path.join(results_sub_folder, params_str + "_score.csv"), index=False)
    os.remove(os.path.join(results_sub_folder, params_str + "_score.txt"))

    title = "RNN STRUCTURE: " + str(NUMBER_OF_BACTERIA) + "-"
    if layer_num == 1:
        title += str(hid_dim) + " -> MLP: " + str(hid_dim) + "-" + str(out_dim)
    if layer_num == 2:
        title += str(hid_dim) + " - " + str(hid_dim) + " -> MLP: " + str(hid_dim) + "-" + str(out_dim)

    title += "  BATCH: " + str(BATCH_SIZE) + " {" + task_id + "}"

    title += "\nLEARNING_RATE: " + str(params["LEARNING_RATE"]) + " OPTIMIZER: " + str(params["OPTIMIZER"]) +\
             " REGULARIZATION: " + str(params["REGULARIZATION"]) + " DROPOUT: " + str(params["DROPOUT"])

    activator.plot_line(title, os.path.join(results_sub_folder, params_str + "_fig"), job=LOSS_PLOT)
    activator.plot_line(title, os.path.join(results_sub_folder, params_str + "_fig"), job=CORR_PLOT)
    dev_avg_loss = np.mean(activator.loss_dev_vec[-10:])
    train_avg_loss = np.mean(activator.loss_train_vec[-10:])

    dev_avg_corr = None
    train_avg_corr = None
    if CALC_CORR:
        # activator.plot_line(title, os.path.join(results_sub_folder, params_str + "_fig"), job=CORR_PLOT)
        dev_avg_corr = np.mean(activator.corr_dev_vec[-10:])
        train_avg_corr = np.mean(activator.corr_train_vec[-10:])
    print(f'train_corr {train_avg_corr}')
    print(f'dev_corr {dev_avg_corr}')
    # activator.scatter_results("Scatter Plot of RNN Results - " + task_id,
    #                           os.path.join(results_sub_folder, params_str + "_Scatter_Plot.png"))
    result_map = {"TRAIN": {"loss": train_avg_loss,
                            "corr": train_avg_corr},
                  "TEST": {"loss": dev_avg_loss,
                            "corr": dev_avg_corr}
                  }
    r_df = pd.DataFrame(result_map)
    r_df.to_csv(os.path.join(results_sub_folder, params_str + "_results_df.csv"))
    return result_map


def run_RNN(X, y, missing_values, name, folder, params, number_of_samples, number_of_time_points, number_of_bacteria, GPU_flag, task_id, unique=5):
    print(name)
    global NUMBER_OF_SAMPLES
    global NUMBER_OF_TIME_POINTS
    global NUMBER_OF_BACTERIA
    return run_rnn_experiment(X, y, missing_values, params, folder, GPU_flag=GPU_flag, task_id=task_id, unique=unique)