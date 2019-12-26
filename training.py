from utils.helper import Helper


class Training:
    def __init__(self, log, args):
        """
        This function is responsible for the parameter initialization, data loading start training process
        :param log:
        :param args:
        """
        self.helper = Helper(log)
        self.log = log

        # model parameter initialization
        self.learning_rate = args['learningRate']
        self.valid = args['valid']
        self.batch_size = args['batchSize']
        self.epoch = args['epoch']
        self.weight_decay = args['weightDecay']

    def set_boilerplates(self):

        """
        set device
        """
        self.helper.set_device()

        """
        download training and validation data
        """
        self.helper.load_cifar_data()

        """
        set train and test data loader
        """
        self.helper.set_data_loader(self.batch_size)

        """
        set models
        """
        self.helper.set_models()

        """
        set loss, optimizer and regularization
        """
        self.helper.set_loss_optimizer(self.weight_decay, self.learning_rate)

        """
        set loss, optimizer and regularization
        """
        if self.valid:
            self.helper.validate()

        self.helper.train(self.epoch)
