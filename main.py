from ttictoc import TicToc
from utils.custom_logging import Logging
from utils.command_parser import CommandParser
from utils.slack_notification import SlackNotification

from training import Training


class Cifar():
    def __init__(self):
        """
        Initialize logger
        """
        self.log = Logging(__name__).get_logger()

        """
        Slack notification registration (can be used to notify when training is finished/ uncomment and use with 
        your slack token)
        """
        # SlackNotification().register(self.log)

        """
        Parse the command line arguments
        """
        arguments = CommandParser().cli_argument_parser(self.log)

        Training(self.log, arguments).set_boilerplates()


if __name__ == "__main__":
    """
        Start of the program
    """

    # try:
    #
    #     t = TicToc()
    #     t.tic()
    #
    Cifar()
    #
    #     t.toc()
    #     print('Time elapsed : {} minutes'.format(t.elapsed/60))
    #
    # except Exception as e:
    #     print('Failed to execute program : Exception : {}'.format(e))
    #     exit(1)
