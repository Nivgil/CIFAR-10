
import logging
import torch
import numpy as np
from torch.autograd import Variable
from bokeh.plotting import figure, curdoc,output_file, show
from dataset import DataSetCifar10
import net_model

class Stats:
    def __init__(self,epochs, data_size, batch_size):
        # iterations = round(data_size/batch_size)
        self.training_loss = torch.FloatTensor(epochs).fill_(0)

    def visualize(self):
        p = figure(plot_width=600, plot_height=600, min_border=10, min_border_left=50,y_axis_type="log",
                   x_axis_label='Iteration', y_axis_label='Loss', title="CIFAR-10 Training Loss")
        p.background_fill_color = "#fafafa"
        # import pdb; pdb.set_trace()
        loss = self.training_loss[:].numpy()
        loss = loss.reshape(loss.size)
        epochs = np.linspace(1,loss.size,loss.size)
        p.line(epochs, loss, color="orange", alpha=0.6)
        output_file('training_loss_CIFAR10.html')
        show(p)


def main():

    filename = 'CIFAR_10_Log'
    logging.basicConfig(format='%(levelname)s:%(message)s', filename='example.log', filemode='w', level=logging.DEBUG)
    # TODO: add logging of simulation

    lr = 1e-3
    momentum = 0.9
    epochs = 100
    batch_size = 10
    data_size = 50000  # training

    stats = Stats(epochs, data_size, batch_size)
    dataset = DataSetCifar10(batch_size)
    trainloader = dataset.get_train()
    model = net_model.Net()
    # model = torch.nn.DataParallel(model) TODO: requires GPU

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    print('Training Model...')
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        # import pdb; pdb.set_trace()
        for iteration, batch in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = batch
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
        if epoch % 10 == 9:  # print every 10 epochs
            print('[Epoch - %d] loss: %.4f' % (epoch + 1, running_loss / data_size))
        stats.training_loss[epoch] = running_loss/data_size
    print('Done')
    import pdb; pdb.set_trace()
    stats.visualize()


if __name__ == '__main__':
    main()