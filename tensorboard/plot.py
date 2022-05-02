import matplotlib.pyplot as plt
import numpy as np
import csv

def load_step(path):
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader)
        data = np.array(list(reader)).astype(float)
        step = data[:, 1]

        return step

def load_loss(path):
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader)
        data = np.array(list(reader)).astype(float)
        step = data[:, 2]

        return step

if __name__ == '__main__':
    step = load_step(path="data/run-task_2.1_basic_model-tag-loss_total_loss.csv")
    basic = load_loss(path="data/run-task_2.1_basic_model-tag-loss_total_loss.csv")
    augmentation = load_loss(path="data/run-task_2.2_augmentation-tag-loss_total_loss.csv")
    retina = load_loss(path="data/run-task_2.3.1_retinanet-tag-loss_total_loss.csv")
    deeper_heads = load_loss(path="data/run-task_2.3.3_deeper_heads-tag-loss_total_loss.csv")

    """
    weight_init = load_loss(path=" ")
    focal_loss = load_loss(path=" ")
    """


    plt.plot(step, basic, label='Basic model')
    plt.plot(step, augmentation, label='Data augmentation')
    plt.plot(step, retina, label='FPN on ResNet')
    plt.plot(step, deeper_heads, label='Deeper convolutional heads')

    """
    plt.plot(step, weight_init, label='Improved weight and bias initialization')
    plt.plot(step, focal_loss, label='Focal loss')
    """

    plt.xlabel('Train step')
    plt.ylabel('Total loss')
    plt.ylim([0.0, 10])
    plt.legend()
    plt.grid()
    plt.savefig('plots/total_loss.png')
    plt.show()