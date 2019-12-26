import os

import torch
import torch.nn as nn
import torch.cuda as cuda
from torch.autograd import Variable
from torchsummary import summary
from torchvision import datasets
from torchvision import transforms
from model.auto_encoder import Autoencoder
from model.cnn_classifier import Classifier


class Helper:
    def __init__(self, log):
        self.log = log
        self.cifar_train = None
        self.cifar_valid = None
        self.device = None
        self.cifar_train_loader = None
        self.cifar_valid_loader = None
        self.aen = None
        self.cnn = None
        self.loss_func_cnn = None
        self.loss_func_aen = None
        self.optimizer_cnn = None
        self.optimizer_aen = None

    def load_cifar_data(self):
        """
        This function is responsible for downloading data and perform necessary transformations on it
        :return: train and validation data
        """
        self.log.info('data downloading stared')
        transform_valid = transforms.Compose([transforms.Resize((32, 32)),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                              ])

        transform_train = transforms.Compose([transforms.Resize((32, 32)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomRotation(10),
                                              transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                              transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                              ])

        self.log.info('performing following transformaions on data -')
        self.log.info('-RandomHorizontalFlip')
        self.log.info('-RandomRotation')
        self.log.info('-RandomAffine shear')
        self.log.info('-ColorJitter : brightness, contrast, saturation')

        self.cifar_train = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
        self.cifar_valid = datasets.CIFAR10('data', train=False, download=True, transform=transform_valid)

        self.log.info('data downloading finished')

        return self.cifar_train, self.cifar_valid

    def set_data_loader(self, batch_size):
        """
        This function returns a data loader for train and validation data, which provide easy access to data with
        desired batch size
        :return: loader
        """

        self.cifar_train_loader = torch.utils.data.DataLoader(self.cifar_train, batch_size=batch_size, shuffle=True,
                                                              num_workers=1)
        self.cifar_valid_loader = torch.utils.data.DataLoader(self.cifar_valid, batch_size=batch_size, shuffle=True,
                                                              num_workers=1)

        return self.cifar_train_loader, self.cifar_valid_loader

    def set_device(self):
        """
        Get device and its specifications
        :return: device
        """
        self.device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def set_models(self):
        """
        set up autoencoder and classifier models
        :return: models
        """

        if self.device == "cuda":
            self.aen = Autoencoder().to(self.device)
            self.cnn = Classifier().to(self.device)
        else:
            self.aen = Autoencoder()
            self.cnn = Classifier()

        self.print_model_summary()

    def print_model_summary(self):
        print("*" * 64)
        print("AUTOENCODER SUMMARY :")
        print("*" * 64)
        summary(self.aen, (3, 32, 32))

        self.log.info("*" * 64)
        self.log.info("AUTOENCODER SUMMARY :")
        self.log.info("*" * 64)
        self.log.info(self.aen)

        print(" ")
        print("CLASSIFIER SUMMARY :")
        print("*" * 64)
        summary(self.cnn, (200, 8, 8))

        self.log.info("*" * 64)
        self.log.info("CLASSIFIER SUMMARY :")
        self.log.info("*" * 64)
        self.log.info(self.cnn)

    def set_loss_optimizer(self, weight_decay, learning_rate):
        """
        - set loss function as Cross Entropy Loss
        - set optimizer as ADAM
        - set regularization as L2 by using WEIGHT DECAY RATE ()
        :param weight_decay:
        :param learning_rate:
        :return:
        """

        # for aen
        # loss function
        self.loss_func_aen = nn.MSELoss().to(self.device)

        # get parameter of cnn to pass to optimizer so that optimizer can modify it
        parameters_aen = list(self.cnn.parameters())

        self.optimizer_aen = torch.optim.Adam(parameters_aen, lr=learning_rate, weight_decay=weight_decay)

        # for cnn
        # loss function
        self.loss_func_cnn = nn.CrossEntropyLoss().to(self.device)

        # get parameter of cnn to pass to optimizer so that optimizer can modify it
        parameters_cnn = list(self.cnn.parameters())

        self.optimizer_cnn = torch.optim.Adam(parameters_cnn, lr=learning_rate, weight_decay=weight_decay)

    def train(self, epoch):
        train_loss_aen = []
        valid_loss_aen = []
        train_loss_cnn = []
        valid_loss_cnn = []

        # running_loss_history_cnn = []
        # running_corrects_history_cnn = []
        # val_running_loss_history_cnn = []
        # val_running_corrects_history_cnn = []

        for i in range(epoch):

            running_loss_aen = 0.0
            running_corrects_aen = 0.0
            val_running_loss_aen = 0.0
            val_running_corrects_aen = 0.0

            running_loss_cnn = 0.0
            running_corrects_cnn = 0.0
            val_running_loss_cnn = 0.0
            val_running_corrects_cnn = 0.0

            # Let's train the model
            total_loss_aen = 0.0
            total_train_iter = 0

            total_loss_cnn = 0.0
            total_iter_cnn = 0

            self.aen.train()
            self.cnn.train()

            for image, label in self.cifar_train_loader:
                image = Variable(image).to(self.device)
                self.optimizer_aen.zero_grad()
                encoder_output, output = self.aen(image)

                loss_aen = self.loss_func_aen(output, image)
                loss_aen.backward()
                self.optimizer_aen.step()

                total_train_iter += 1
                total_loss_aen += loss_aen.data.item()

                # ------ cnn

                encoder_output = Variable(encoder_output).to(self.device)
                label = Variable(label).to(self.device)

                self.optimizer_cnn.zero_grad()
                output_cnn = self.cnn(encoder_output)

                loss_cnn = self.loss_func_cnn(output_cnn, label)
                loss_cnn.backward()
                self.optimizer_cnn.step()

                total_iter_cnn += 1
                total_loss_cnn += loss_cnn.data.item()

                _, preds = torch.max(output_cnn, 1)
                #         running_loss_cnn += loss_cnn.item()
                running_corrects_cnn += torch.sum(preds == label.data)

            #     epoch_loss_cnn = running_loss_cnn/len(cifar10_train_loader)
            epoch_acc_cnn = running_corrects_cnn.float() / len(self.cifar_train_loader)
            #     running_loss_history_cnn.append(epoch_loss_cnn)
            #     running_corrects_history_cnn.append(epoch_acc_cnn)

            total_val_loss_aen = 0.0
            total_val_iter = 0
            total_val_loss_cnn = 0.0
            total_val_iter_cnn = 0
            self.aen.eval()
            self.cnn.eval()

            for image, label in self.cifar_valid_loader:
                image = Variable(image).to(self.device)

                encoder_output, output = self.aen(image)
                loss = self.loss_func_aen(output, image)

                total_val_iter += 1
                total_val_loss_aen += loss.data.item()

                encoder_output = Variable(encoder_output).to(self.device)
                label = Variable(label).to(self.device)

                output_cnn = self.cnn(encoder_output)

                loss_cnn = self.loss_func_cnn(output_cnn, label)

                total_val_iter_cnn += 1
                total_val_loss_cnn += loss_cnn.data.item()

                _, val_preds = torch.max(output_cnn, 1)
                #         val_running_loss_cnn += loss_cnn.item()
                val_running_corrects_cnn += torch.sum(val_preds == label.data)

            train_loss_aen.append(total_loss_aen / total_train_iter)
            valid_loss_aen.append(total_val_loss_aen / total_val_iter)

            train_loss_cnn.append(total_loss_cnn / total_train_iter)
            valid_loss_cnn.append(total_val_loss_cnn / total_val_iter)

            #     val_epoch_loss_cnn = val_running_loss_cnn/len(self.cifar_valid_loader)
            val_epoch_acc_cnn = val_running_corrects_cnn.float() / len(self.cifar_valid_loader)
            #     val_running_loss_history_cnn.append(val_epoch_loss_cnn)
            #     val_running_corrects_history_cnn.append(val_epoch_acc_cnn)
            print('epoch :', (i))
            print('AEN : train loss: {:.4f} '.format(total_loss_aen / len(self.cifar_train_loader)))
            print('AEN : valid loss: {:.4f} '.format(total_val_loss_aen / len(self.cifar_valid_loader)))

            print('CNN : train loss: {:.4f}, train acc {:.4f} '.format(total_loss_cnn / len(self.cifar_train_loader),
                                                                       epoch_acc_cnn.item()))
            print(
                'CNN : valid loss: {:.4f}, valid acc {:.4f} '.format(total_val_loss_cnn / len(self.cifar_valid_loader),
                                                                     val_epoch_acc_cnn.item()))
            print('==========================================================')

            # Let's record the validation loss
        print('Training finished successfully')

        # save models
        self.save_model()

    def save_model(self):
        print('Saving models...')
        self.log.info('Saving models..')

        if not os.path.exists('pickle'):
            os.mkdir('pickle')

        torch.save(self.aen.state_dict(), "pickle/autoencoder.pkl")
        print('Saved autoencoder successfully : pickle/autoencoder.pkl')
        self.log.info('Saved autoencoder successfully : pickle/autoencoder.pkl')

        torch.save(self.cnn.state_dict(), "pickle/classifier.pkl")
        print('Saved classifier successfully : pickle/classifier.pkl')
        self.log.info('Saved classifier successfully : pickle/classifier.pkl')

    def validate(self):
        print("Loading checkpoint...")
        self.aen.load_state_dict(
            torch.load("../pickle/autoencoder.pkl", map_location=lambda storage, loc: storage))
        dataiter = iter(self.cifar_valid_loader)
        images, labels = dataiter.next()
        print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(16)))
        imshow(torchvision.utils.make_grid(images))

        # images = Variable(images.cuda())

        decoded_imgs = autoencoder(images)[1]
        imshow(torchvision.utils.make_grid(decoded_imgs.data))

        exit(0)
