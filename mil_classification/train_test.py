import torch
import torch.nn as nn
import numpy as np

from .arg_parser import get_parser
from . import data_manager, models, utils
from .constants import FEATURE_KEYS

class Runner(object):
    def __init__(self, input_size, learning_rate):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = models.AdaptivePoolingClassifier(self.device, input_size).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def run(self, dataloader, mode='train'):
        self.model.train() if mode is 'train' else self.model.eval()

        epoch_loss = 0
        prediction_list = []
        answer_list = []

        self.model = self.model.float()
        for _, (x_list, y, name, measure_list) in enumerate(dataloader):
            x_list = x_list.to(self.device, dtype=torch.float)
            prediction = self.model(x_list)
        
            y = y.to(self.device)
            
            loss = self.criterion(prediction, y)

            if mode is 'train':
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            epoch_loss += prediction.size(0) * loss.item()

            prediction_list.append(prediction.tolist())
            answer_list.append(y.tolist())

        epoch_loss = epoch_loss / len(dataloader.dataset)
        total_result, total_accuracy = utils.get_result(prediction_list, answer_list)

        return epoch_loss, total_accuracy, total_result


def main():
    torch.manual_seed(1234)

    p = get_parser()
    args = p.parse_args()
    feature_keys = FEATURE_KEYS

    train_loader, valid_loader, test_loader = data_manager.get_dataloader(args.path, args.data_name, feature_keys, args.batch_size)

    runner = Runner(len(feature_keys), args.learning_rate)

    print('Training : ')
    for epoch in range(args.num_epoch):
        train_loss, train_acc, train_result = runner.run(train_loader, mode='train')
        valid_loss, valid_acc, valid_result = runner.run(valid_loader, mode='eval')
        print("[Epoch %d/%d] [Train Loss: %.4f] [Train Acc: %.4f%%] [Valid Loss: %.4f] [Valid Acc: %.4f%%]" %
              (epoch + 1, args.num_epoch, train_loss, train_acc, valid_loss, valid_acc))
    
    _, test_acc, test_result = runner.run(test_loader, mode='test')
    print("Training Finished")
    print("Training Accuracy: %.4f%%" % train_acc)
    utils.print_total_result(train_result)

    print("Validation Accuracy: %.4f%%" % valid_acc)
    utils.print_total_result(valid_result)

    print("Test Accuracy: %.4f%%" % test_acc)
    utils.print_total_result(test_result)


if __name__ == "__main__":
    main()
