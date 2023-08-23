from matplotlib import pyplot as plt


class HistLogger:
    @staticmethod
    def saveHist(label, history):
        # "Loss"
        plt.clf()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
        plt.savefig(label)