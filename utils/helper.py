import os

import numpy as np
import torch
import torch.nn as nn
import torch.cuda as cuda
import torchvision
from torch.autograd import Variable
from torchsummary import summary
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Subset
from torch.utils.data import Dataset
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

        self.input_path = os.path.join(os.path.dirname(__file__), '../data/')
        self.output_path = os.path.join(os.path.dirname(__file__), '../output/')
        self.pickle_path = os.path.join(os.path.dirname(__file__), '../pickle/')

        self.aen_file = os.path.join(self.pickle_path, 'autoencoder.pkl')
        self.cnn_file = os.path.join(self.pickle_path, 'classifier.pkl')

        self.original_img = os.path.join(self.output_path, 'original_image.png')
        self.reconstructed_img = os.path.join(self.output_path, 'reconstructed_image.png')

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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

        self.check_dir(self.input_path)

        self.cifar_valid = datasets.CIFAR10(self.input_path, train=False, download=True, transform=transform_valid)
        self.cifar_train = CifarCustomTrainDataset(self.input_path, transform_train)
        print("\nAfter applying constraints")
        print("\ntrain data size : ", len(self.cifar_train))
        print("valid data size : ", len(self.cifar_valid))
        print("truck size      : ", len(self.cifar_train.truck_indices))
        print("deer size       : ", len(self.cifar_train.deer_indices))
        print("bird size       : ", len(self.cifar_train.bird_indices))
        print("other size      : ", len(self.cifar_train.other_indices))

        self.log.info("train data size : {}".format(len(self.cifar_train)))
        self.log.info("valid data size : {}".format(len(self.cifar_valid)))
        self.log.info("truck size      : {}".format(len(self.cifar_train.truck_indices)))
        self.log.info("deer size       : {}".format(len(self.cifar_train.deer_indices)))
        self.log.info("bird size       : {}".format(len(self.cifar_train.bird_indices)))
        self.log.info("other size      : {}".format(len(self.cifar_train.other_indices)))

        self.log.info('data downloading finished')

        print('\ncreating sampler for train data to deal with imbalance dataset')
        self.log.info('\ncreating sampler for train data to deal with imbalance dataset')
        self.sampler = ImbalancedDatasetSampler(self.cifar_train)

        print('\nsampler for train data to deal with imbalance dataset is completed')
        self.log.info('\ncreating sampler for train data to deal with imbalance dataset is completed')

        return self.cifar_train, self.cifar_valid

    def set_data_loader(self, batch_size):
        """
        This function returns a data loader for train and validation data, which provide easy access to data with
        desired batch size
        :return: loader
        """

        self.cifar_train_loader = torch.utils.data.DataLoader(self.cifar_train, batch_size=batch_size, shuffle=False,
                                                              num_workers=1, sampler=self.sampler)
        self.cifar_valid_loader = torch.utils.data.DataLoader(self.cifar_valid, batch_size=batch_size, shuffle=True,
                                                              num_workers=1)

        return self.cifar_train_loader, self.cifar_valid_loader

    def set_device(self):
        """
        Get device and its specifications
        :return: device
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(self.device) == 'cuda:0':
            self.device_info()

    def set_models(self):
        """
        set up autoencoder and classifier models
        :return: models
        """

        self.aen = Autoencoder().to(self.device)
        self.cnn = Classifier().to(self.device)

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

        self.optimizer_aen = torch.optim.Adam(parameters_aen, lr=learning_rate)

        # for cnn
        # loss function
        self.loss_func_cnn = nn.CrossEntropyLoss().to(self.device)

        # get parameter of cnn to pass to optimizer so that optimizer can modify it
        parameters_cnn = list(self.cnn.parameters())

        self.optimizer_cnn = torch.optim.Adam(parameters_cnn, lr=learning_rate, weight_decay=weight_decay)

    def device_info(self):
        """
        Parse output of nvidia-smi into a python dictionary.
        This is very basic!
        """

        import subprocess
        try:
            sp = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            out_str = sp.communicate()
            out_list = out_str[0].decode("utf-8").split('\n')

            out_dict = {}

            for item in out_list:
                try:
                    key, val = item.split(':')
                    key, val = key.strip(), val.strip()
                    out_dict[key] = val
                except:
                    pass

            print('\nGPU FOUND with following specification -')
            print(' -Product Name : {}'.format(out_dict.get('Product Name')))
            print(' -Product Brand : {}'.format(out_dict.get('Product Brand')))
            print(' -Used GPU Memory : {}'.format(out_dict.get('Used GPU Memory')))

            self.log.info('GPU FOUND with following specification -')
            self.log.info(' -Product Name : {}'.format(out_dict.get('Product Name')))
            self.log.info(' -Product Brand : {}'.format(out_dict.get('Product Brand')))
            self.log.info(' -Used GPU Memory : {}'.format(out_dict.get('Used GPU Memory')))
        except Exception as e:
            print('Unable to print GPU info : {}'.format(str(e)))
            self.log.info('Unable to print GPU info : {}'.format(str(e)))

    def train(self, epoch):

        print('Training Started ...')
        self.log.info("Training Started ...")
        train_loss_aen = []
        valid_loss_aen = []
        train_loss_cnn = []
        valid_loss_cnn = []
        train_acc_cnn = []
        valid_acc_cnn = []

        for i in range(epoch):

            epoch_train_loss_aen = 0.0
            epoch_train_loss_cnn = 0.0
            epoch_train_iter = 0

            epoch_val_loss_aen = 0.0
            epoch_val_loss_cnn = 0.0
            epoch_val_iter = 0

            running_train_corrects_cnn = 0.0
            running_val_corrects_cnn = 0.0

            total_train = 0
            total_val = 0

            self.aen.train()
            self.cnn.train()

            for image, label in self.cifar_train_loader:
                # autoencoder training
                self.optimizer_aen.zero_grad()

                image = image.to(self.device)
                encoder_output, output = self.aen(image)

                loss_aen = self.loss_func_aen(output, image)

                loss_aen.backward(retain_graph=True)
                self.optimizer_aen.step()

                total_train += label.size(0)
                epoch_train_iter += 1
                epoch_train_loss_aen += loss_aen.data.item()

                # cnn training
                self.optimizer_cnn.zero_grad()
                encoder_output = encoder_output.to(self.device)
                output_cnn = self.cnn(encoder_output)

                label = label.to(self.device)
                loss_cnn = self.loss_func_cnn(output_cnn, label)

                loss_cnn.backward()
                self.optimizer_cnn.step()

                # count number of correct predictions
                _, preds = torch.max(output_cnn, 1)
                running_train_corrects_cnn += torch.sum(preds == label.data)

                epoch_train_loss_cnn += loss_cnn.data.item()

            self.aen.eval()
            self.cnn.eval()

            for image, label in self.cifar_valid_loader:
                image = image.to(self.device)

                encoder_output, output = self.aen(image)
                loss = self.loss_func_aen(output, image)

                total_val += label.size(0)
                epoch_val_iter += 1
                epoch_val_loss_aen += loss.data.item()

                # cnn training
                encoder_output = encoder_output.to(self.device)
                output_cnn = self.cnn(encoder_output)
                label = label.to(self.device)

                _, preds = torch.max(output_cnn, 1)
                running_val_corrects_cnn += torch.sum(preds == label.data)

                loss_cnn = self.loss_func_cnn(output_cnn, label)

                epoch_val_loss_cnn += loss_cnn.data.item()

            print('aen loss epoch [{}/{}], train:{:.4f},  valid:{:.4f}'.format(i + 1, epoch,
                                                                               epoch_train_loss_aen/ epoch_train_iter,
                                                                               epoch_val_loss_aen / epoch_val_iter))
            print('cnn loss epoch [{}/{}], train:{:.4f},  valid:{:.4f}'.format(i + 1, epoch,
                                                                               epoch_train_loss_cnn / epoch_train_iter,
                                                                               epoch_val_loss_cnn / epoch_val_iter))
            print('cnn acc  epoch [{}/{}], train:{:.2f}%, valid:{:.2f}%'.format(i + 1,
                                                                                epoch,
                                                                                running_train_corrects_cnn.float() / total_train * 100,
                                                                                running_val_corrects_cnn.float() / total_val * 100))

            train_loss_aen.append(epoch_train_loss_aen / epoch_train_iter)
            valid_loss_aen.append(epoch_val_loss_aen / epoch_val_iter)
            train_loss_cnn.append(epoch_train_loss_cnn / epoch_train_iter)
            valid_loss_cnn.append(epoch_val_loss_cnn / epoch_val_iter)
            train_acc_cnn.append(running_train_corrects_cnn.float() / total_train * 100)
            valid_acc_cnn.append(running_val_corrects_cnn.float() / total_val * 100)

            print('-' * 60)

        print('Training finished successfully')

        # save models
        self.save_model()

    def save_model(self):
        print('Saving models...')
        self.log.info('Saving models..')

        self.check_dir(self.pickle_path)

        torch.save(self.aen.state_dict(), self.aen_file)
        print('Saved autoencoder successfully : {}'.format(self.aen_file))
        self.log.info('Saved autoencoder successfully : {}'.format(self.aen_file))

        torch.save(self.cnn.state_dict(), self.cnn_file)
        print('Saved classifier successfully : {}'.format(self.cnn_file))
        self.log.info('Saved classifier successfully : {}'.format(self.cnn_file))

    def validate(self):
        print("Loading checkpoint...")
        self.aen.load_state_dict(torch.load(self.aen_file, map_location=lambda storage, loc: storage))
        dataiter = iter(self.cifar_valid_loader)
        images, labels = dataiter.next()

        self.check_dir(self.output_path)

        torchvision.utils.save_image(images[:10], self.original_img, nrow=2, scale_each=True)

        images = images.to(self.device)

        features, output = self.aen(images)
        pred_labels = self.cnn(features)
        _, val_preds = torch.max(pred_labels, 1)

        print('\nGroundTruth: ', ' '.join('%10s' % self.classes[labels[j]] for j in range(10)))
        print('Prediciton : ', ' '.join('%10s' % self.classes[val_preds[j].item()] for j in range(10)))

        torchvision.utils.save_image(output[:10], self.reconstructed_img, nrow=2, scale_each=True)
        print('')

        print('Input image path : {}'.format(self.original_img))
        print('Reconstructed image path : {}'.format(self.reconstructed_img))
        # print("Predicted Labels : {}".format(labels))
        #
        self.log.info('Input image path : {}'.format(self.original_img))
        self.log.info('Reconstructed image path : {}'.format(self.reconstructed_img))
        # self.log.info("Predicted Labels : {}".format(labels))

        exit(0)

    def check_dir(self, path):
        if not os.path.exists(path):
            print(path)
            os.mkdir(path)

    def im_convert(self, file_name, tensor):
        image = tensor.cpu().clone().detach().numpy()
        image = image.transpose(1, 2, 0)
        image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
        image = image.clip(0, 1)

        import matplotlib.pyplot as plt

        plt.imsave(file_name, image)

        return image


class CifarCustomTrainDataset(Dataset):

    def __init__(self, input_path, transforms=None):
        '''
        Create a dataset with only 50% of bird, deer and truck classes, remaining classes with 100%
        '''
        self.transforms = transforms
        self.cifar10_train = datasets.CIFAR10(input_path, train=True, download=True)

        self.truck_indices, self.deer_indices, self.bird_indices, self.other_indices = [], [], [], []
        truck_idx, deer_idx, bird_idx = self.cifar10_train.class_to_idx['truck'], self.cifar10_train.class_to_idx[
            'deer'], self.cifar10_train.class_to_idx['bird']

        for i in range(len(self.cifar10_train)):
            current_class = self.cifar10_train[i][1]
            if current_class == truck_idx:
                self.truck_indices.append(i)
            elif current_class == deer_idx:
                self.deer_indices.append(i)
            elif current_class == bird_idx:
                self.bird_indices.append(i)
            else:
                self.other_indices.append(i)

        self.truck_indices = self.truck_indices[:int(0.5 * len(self.truck_indices))]
        self.deer_indices = self.deer_indices[:int(0.5 * len(self.deer_indices))]
        self.bird_indices = self.bird_indices[:int(0.5 * len(self.bird_indices))]
        self.cifar_10 = Subset(self.cifar10_train,
                               self.truck_indices + self.deer_indices + self.bird_indices + self.other_indices)

    def __len__(self):
        return len(self.cifar_10)

    def __getitem__(self, idx):
        data = self.cifar_10[idx]
        if self.transforms is not None:
            data = (self.transforms(data[0]), data[1])
        return data

        # image counter function for weight calculation

    def get_label(self, idx):
        data = self.cifar_10[idx]
        return data[1]


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.get_label(idx)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
