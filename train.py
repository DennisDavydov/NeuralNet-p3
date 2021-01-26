import MNIST_loader as MNIST
import Network

training_data, validation_data, test_data = MNIST.load_data_wrapper()
sizes = [784, 60, 10]

net = Network.Network(sizes)

net.SGD(training_data, 20, 1000, 3.0, test_data = test_data)

