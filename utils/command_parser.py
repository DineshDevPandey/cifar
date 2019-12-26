import sys
import argparse
import utils.config_parser
from utils.config_parser import ConfParser

class CommandParser(object):
    def __init__(self):
        self.config = ConfParser()

    def cli_argument_parser(self, log):

        """
        Command line argument parsing function
        :return:
        """

        try:
            parser = argparse.ArgumentParser(prog='cifar', description='cifar image classification with autoencoder')

            parser.add_argument('-valid',
                                '--valid',
                                type=int,
                                dest='valid',
                                default=self.config.parser.get('CIFAR', 'valid'),
                                help='train and validation: 0, validation only: 1')

            parser.add_argument('-epoch',
                                '--epoch',
                                type=int,
                                dest='epoch',
                                default=self.config.parser.get('CIFAR', 'epoch'),
                                help='Number of epochs')

            parser.add_argument('-batchSize',
                                '--batchSize',
                                type=int,
                                dest='batchSize',
                                default=self.config.parser.get('CIFAR', 'batchSize'),
                                help='batch size')

            parser.add_argument('-weightDecay',
                                '--weightDecay',
                                type=float,
                                dest='weightDecay',
                                default=self.config.parser.get('CIFAR', 'weightDecay'),
                                help='weight decay for L2 normalization')

            parser.add_argument('-lr',
                                '--learningRate',
                                type=float,
                                dest='learningRate',
                                default=self.config.parser.get('CIFAR', 'learningRate'),
                                help='learning rate')

            args = parser.parse_args()

            log.info("Arguments : = {}".format(args))
            print("Arguments : = {}".format(args))
            return dict(args._get_kwargs())

        except Exception as e:
            log.error("Error while parsing CLI Arguments : {}".format(e))
            raise e

#
# if __name__ == "__main__":
#     aa = CommandParser()
#     print(aa.cli_argument_parser())
