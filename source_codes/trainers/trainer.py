import copy
import logging

import torch
import torch.optim as optim


class Trainer(object):
    def __init__(self,
                 model=None,
                 train_dataloader=None,
                 valid_dataloader=None,
                 train_epochs=1000,
                 valid_epochs=10,
                 learning_rate=0.5,
                 lr_decay=0,
                 weight_decay=0,
                 scheduler=None,
                 optimization_method='sgd',
                 saved_epochs=None,
                 checkpoint_dir=None,
                 working_threads=0,
                 log_interval=1,
                 use_gpu=True,
                 loss_func=None
                 ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.train_epochs = train_epochs
        self.valid_epochs = valid_epochs
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.optmization_method = optimization_method
        self.saved_epochs = saved_epochs
        self.checkpoint_dir = checkpoint_dir
        self.working_threads = working_threads
        self.log_interval = log_interval
        self.loss_func = loss_func
        self.optimizer = None
        self.use_gpu = use_gpu
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

        self.best_valacc = 0.0
        self.best_epoch = 0.0

    def train_one_step(self,):
        self.model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        nsamples = 0

        for inputs, labels in self.train_dataloader:

            if self.use_gpu:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

            nsamples += inputs.shape[0]

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(mode=True):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.loss_func(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        if self.scheduler is not None:
            self.scheduler.step()

        epoch_loss = running_loss / nsamples
        epoch_acc = running_corrects.double() / nsamples

        return epoch_loss, epoch_acc

    def valid_one_step(self, ):
        self.model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        nsamples = 0
        for inputs, labels in self.valid_dataloader:
            if self.use_gpu:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

            nsamples += inputs.shape[0]

            # zero the parameter gradients
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.loss_func(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / nsamples
        epoch_acc = running_corrects.double() / nsamples

        return epoch_loss, epoch_acc

    def save_model(self, save_path):
        """Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.

        Args:
            save_path (str): path where the model is saved
        """

        state = {
            'state_dict': self.model.state_dict(),
            'best_valacc': self.best_valacc,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, save_path)

    def load_model(self, load_path):
        """Function to load a saved model

        Args:
            load_path (_type_): path to the saved model
        """
        state = torch.load(load_path)
        state_dict = state['state_dict']
        self.best_valacc = state['best_valacc']
        self.best_epoch = state['best_epoch']

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])

    def run(self):
        # Check use GPU or CPU
        if self.use_gpu:
            self.model.cuda()

        if self.optimizer != None:
            pass
        elif self.optmization_method == "Adagrad" or self.optmization_method == "adagrad":
            self.optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=self.learning_rate,
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,
            )
        elif self.optmization_method == "Adadelta" or self.optmization_method == "adadelta":
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optmization_method == "Adam" or self.optmization_method == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        best_model_wts = copy.deepcopy(self.model.state_dict())

        best_acc = 0.0

        best_epoch = 0.0

        losses, accuracies = dict(train=[], val=[]), dict(train=[], val=[])

        for epoch in range(self.train_epochs):
            if self.log_interval is not None and epoch % self.log_interval == 0:
                logging.info(
                    'Epoch {}/{}'.format(epoch, self.train_epochs - 1))
                logging.info('-' * 10)

            if epoch % self.valid_epochs != 0:
                phase = 'train'
                epoch_loss, epoch_acc = self.train_one_step()
                losses[phase].append(epoch_loss)
                accuracies[phase].append(epoch_acc)

                if self.log_interval is not None and epoch % self.log_interval == 0:
                    logging.info('{} Loss: {:.4f} Acc: {:.5f}'.format(
                        phase, epoch_loss, epoch_acc))

            else:
                phase = 'val'

                epoch_loss, epoch_acc = self.valid_one_step()
                losses[phase].append(epoch_loss)
                accuracies[phase].append(epoch_acc)

                if self.log_interval is not None and epoch % self.log_interval == 0:
                    logging.info('{} Loss: {:.4f} Acc: {:.5f}'.format(
                        phase, epoch_loss, epoch_acc))

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(self.model.state_dict())

        # print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logging.info('Best val Acc: {:.5f}'.format(best_acc))

        self.model.load_state_dict(best_model_wts)
        self.best_valacc = best_acc
        self.best_epoch = best_epoch

        return self.model, losses, accuracies
