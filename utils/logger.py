from matplotlib import pyplot as plt
import os


class Logger:
    def __init__(self):
        self.loss_train = []
        self.loss_val = []

        self.acc_train = []
        self.acc_val = []

        self.best_acc = 0
        self.best_loss = float("inf")

    def get_logs(self):
        return self.loss_train, self.loss_val, self.acc_train, self.acc_val, self.best_acc, self.best_loss

    def restore_logs(self, logs):
        if len(logs) == 4:
            self.loss_train, self.loss_val, self.acc_train, self.acc_val = logs
        else:
            self.loss_train, self.loss_val, self.acc_train, self.acc_val, self.best_acc, self.best_loss = logs

    def save_plt(self, hps):
        loss_path = os.path.join(hps['model_save_dir'], 'loss.jpg')
        acc_path = os.path.join(hps['model_save_dir'], 'acc.jpg')

        plt.figure()
        plt.plot(self.acc_train, 'g', label='Training Acc')
        plt.plot(self.acc_val, 'b', label='Validation Acc')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.legend()
        plt.grid()
        plt.savefig(acc_path)

        plt.figure()
        plt.plot(self.loss_train, 'g', label='Training Loss')
        plt.plot(self.loss_val, 'b', label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.savefig(loss_path)
