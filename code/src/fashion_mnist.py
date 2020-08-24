from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets

from torch.utils.data import TensorDataset

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt 

import numpy as np

import pickle

class Student(nn.Module):
    def __init__(self, input_dim = 784, output_dim = 10, device = 'cpu'):
        super(Student, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.linear = nn.Linear(input_dim, output_dim)
        
        self.to(device)
        
    def forward(self, input):
        out = self.linear(input)
        return out
    
class Teacher(nn.Module):
    def __init__(self, device = 'cpu'):
        super(Teacher, self).__init__()
        
        output_dim = 10

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size = 5)
        self.conv2 = nn.Conv2d(3, 9, kernel_size = 5)
        self.linear1 = nn.Linear(9*4*4, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, output_dim)
        
        self.to(device)
        
    def forward(self, input):
        out = self.conv1(input)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size = 2)
        out = self.conv2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size = 2)
        
        out = out.view(out.shape[0], -1)
        
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        return out
    
    
def train_student_without_teacher(train_data, test_data, epochs=100, meaning=5):
    list_of_student_models = []
    list_of_acc_train = []
    list_of_acc_test = []
    list_of_losses_train = []
    list_of_losses_test = []

    for tryes in tqdm(range(meaning)):

        student = Student()

        optimazir = optim.Adam(student.parameters())
        loss_function = torch.nn.CrossEntropyLoss()

        iterator = tqdm(range(epochs), leave=False)
        iterator.set_postfix_str('epoch 0; loss: train nan test nan; acc: train nan test nan')

        list_of_train_loss = []
        list_of_test_loss = []
        list_of_train_acc = []
        list_of_test_acc = []

        for i in iterator:
            dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

            epoch_loss = 0
            epoch_true = 0
            for x, y in dataloader:
                optimazir.zero_grad()

                predict = student(x)

                loss = loss_function(predict, y)

                loss.backward()

                optimazir.step()

                epoch_loss += loss.item()*len(y)

                epoch_true += (torch.argmax(predict, axis=1) == y).sum().item()

            testloader = DataLoader(test_data, batch_size=64, shuffle=False)
            test_loss = 0
            test_true = 0
            for x, y in testloader:
                predict = student(x)
                loss = loss_function(predict, y)
                test_loss += loss.item()*len(y)

                test_true += (torch.argmax(predict, axis=1) == y).sum().item()

            list_of_train_loss.append(epoch_loss/len(train_data))
            list_of_test_loss.append(test_loss/len(test_data))

            list_of_train_acc.append(epoch_true/len(train_data))
            list_of_test_acc.append(test_true/len(test_data))

            iterator.set_postfix_str(
                'epoch {}; loss: train {} test {}; acc: train {} test {}'.format(
                    i, 
                    round(list_of_train_loss[-1], 2), 
                    round(list_of_test_loss[-1], 2), 
                    round(list_of_train_acc[-1], 2), 
                    round(list_of_test_acc[-1], 2)))

        list_of_losses_train.append(list_of_train_loss)
        list_of_losses_test.append(list_of_test_loss)

        list_of_acc_train.append(list_of_train_acc)
        list_of_acc_test.append(list_of_test_acc)

        list_of_student_models.append(student)
        
    DICT = dict()
    DICT['list_of_student_models'] = list_of_student_models
    DICT['list_of_acc_train'] = list_of_acc_train
    DICT['list_of_acc_test'] = list_of_acc_test
    DICT['list_of_losses_train'] = list_of_losses_train
    DICT['list_of_losses_test'] = list_of_losses_test

    return DICT
    
    
def train_teacher(train_data_cnn, test_data_cnn, epochs = 20):
    teacher = Teacher()
    
    optimazir = optim.Adam(teacher.parameters())
    loss_function = torch.nn.CrossEntropyLoss()

    epochs = 20
    iterator = tqdm(range(epochs))
    iterator.set_postfix_str('epoch 0; loss: train nan test nan; acc: train nan test nan')

    for i in iterator:
        dataloader = DataLoader(train_data_cnn, batch_size=64, shuffle=True)
        epoch_loss = 0
        epoch_true = 0
        teacher.train()
        for x, y in dataloader:
            optimazir.zero_grad()

            predict = teacher(x)

            loss = loss_function(predict, y)

            loss.backward()

            optimazir.step()

            epoch_loss += loss.item()*len(y)

            epoch_true += (torch.argmax(predict, axis=1) == y).sum().item()

        testloader = DataLoader(test_data_cnn, batch_size=64, shuffle=False)
        test_loss = 0
        test_true = 0
        teacher.eval()
        for x, y in testloader:
            predict = teacher(x)
            loss = loss_function(predict, y)

            test_loss += loss.item()*len(y)

            test_true += (torch.argmax(predict, axis=1) == y).sum().item()

        iterator.set_postfix_str(
            'epoch {}; loss: train {} test {}; acc: train {} test {}'.format(
                i, 
                round(epoch_loss/len(train_data_cnn), 2),
                round(test_loss/len(test_data_cnn), 2),
                round(epoch_true/len(train_data_cnn), 2),
                round(test_true/len(test_data_cnn), 2)))
        
        
    DICT['model'] = teacher

    return DICT
    
    
def train_student_with_teacher(teacher,
                               train_data, 
                               test_data, 
                               train_data_cnn, 
                               test_data_cnn, 
                               epochs=100, 
                               meaning=5, 
                               T=2, 
                               lamb=0.25):
    
    dataloader = DataLoader(train_data_cnn, batch_size=len(train_data_cnn), shuffle=False)
    for x, y in dataloader:
        S = torch.softmax(teacher(x)/T, axis=1).detach()
        
    for x, y in DataLoader(train_data, batch_size=len(train_data), shuffle=False):
        pass
    
    all_train_data = TensorDataset(x, y, S)
    
    
    list_of_student_models_dist = []
    list_of_acc_train_dist = []
    list_of_acc_test_dist = []
    list_of_losses_train_dist = []
    list_of_losses_test_dist = []

    epochs = 100
    meaning = 5

    for tryes in tqdm(range(meaning)):

        student = Student()

        optimazir = optim.Adam(student.parameters())
        loss_function = torch.nn.CrossEntropyLoss()

        iterator = tqdm(range(epochs), leave=False)
        iterator.set_postfix_str('epoch 0; loss: train nan test nan; acc: train nan test nan')

        list_of_train_loss = []
        list_of_test_loss = []
        list_of_train_acc = []
        list_of_test_acc = []

        for i in iterator:
            dataloader = DataLoader(all_train_data, batch_size=64, shuffle=True)

            epoch_loss = 0
            epoch_true = 0
            for x, y, s in dataloader:
                optimazir.zero_grad()

                predict = student(x)
                log_soft_pred = torch.log(torch.softmax(predict/T, axis=1))

                loss = (1-lamb)*loss_function(predict, y) \
                                 - lamb*(s*log_soft_pred).mean() \
                                 - lamb*(log_soft_pred + torch.log(-log_soft_pred)).mean()

                loss.backward()

                optimazir.step()

                epoch_loss += loss.item()*len(y)

                epoch_true += (torch.argmax(predict, axis=1) == y).sum().item()

            testloader = DataLoader(test_data, batch_size=64, shuffle=False)
            test_loss = 0
            test_true = 0
            for x, y in testloader:
                predict = student(x)
                loss = loss_function(predict, y)
                test_loss += loss.item()*len(y)

                test_true += (torch.argmax(predict, axis=1) == y).sum().item()

            list_of_train_loss.append(epoch_loss/len(train_data))
            list_of_test_loss.append(test_loss/len(test_data))

            list_of_train_acc.append(epoch_true/len(train_data))
            list_of_test_acc.append(test_true/len(test_data))

            iterator.set_postfix_str(
                'epoch {}; loss: train {} test {}; acc: train {} test {}'.format(
                    i, 
                    round(list_of_train_loss[-1], 2), 
                    round(list_of_test_loss[-1], 2), 
                    round(list_of_train_acc[-1], 2), 
                    round(list_of_test_acc[-1], 2)))

        list_of_losses_train_dist.append(list_of_train_loss)
        list_of_losses_test_dist.append(list_of_test_loss)

        list_of_acc_train_dist.append(list_of_train_acc)
        list_of_acc_test_dist.append(list_of_test_acc)

        list_of_student_models_dist.append(student)
        
    DICT = dict()
    DICT['list_of_student_models_dist'] = list_of_student_models_dist
    DICT['list_of_losses_train_dist'] = list_of_losses_train_dist
    DICT['list_of_losses_test_dist'] = list_of_losses_test_dist
    DICT['list_of_acc_train_dist'] = list_of_acc_train_dist
    DICT['list_of_acc_test_dist'] = list_of_acc_test_dist

    return DICT