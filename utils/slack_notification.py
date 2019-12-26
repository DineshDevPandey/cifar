import os
import slack
import atexit


class SlackNotification(object):

    def __init__(self):
        """
        Class responsible for posting message to slack, once program will terminate normally or abnormally
        :param master_id:
        """

        self.client = slack.WebClient(token='YOUR TOKEN')

    def slack_message(self):
        """
        Method for sending message to slack
        :return:
        """
        response = self.client.chat_meMessage(
            channel='#deep',
            text="Training process is finished, please check logs.")
        assert response["ok"]

    def register(self, log):
        """
        Method to register this function with atexit
        Whenever program will terminate this function will get invoked
        :return:
        """
        atexit.register(self.slack_message)
        log.info('Slack notification registered')

