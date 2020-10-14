import torch
import numpy as np
import pandas as pd
import time

from .constants import NUM_EPOCH, FEATURE_KEYS, LEARNING_RATE, TEST_FEATURE_KEYS
from . import data_manager, models, utils
from .arg_parser import get_parser

class Runner(object):
    def __init__(self, input_size):
        self.model = models.SimpleClassifier(input_size)
        self.model = self.model.double()
        
        #self.criterion = torch.nn.MultiMarginLoss()
        self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE)

        self.device = torch.device("cpu")
        

    def run(self, dataloader, mode='train'):
        self.model.train() if mode is 'train' else self.model.eval()

        epoch_loss = 0
        prediction_list = []
        answer_list = []
        for batch, (x, y) in enumerate(dataloader):
            x = x.to(self.device)
            y = y.to(self.device, dtype=torch.long)
            #y = y.view(y.size(1))

            prediction = self.model(x)
            
            loss = self.criterion(prediction.view(1, prediction.size(0)), y)
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
    #torch.manual_seed(1234)
    
    p = get_parser()
    args = p.parse_args()
    feature_keys = TEST_FEATURE_KEYS

    train_loader, valid_loader, test_loader = data_manager.get_dataloader(args.path, args.name, feature_keys)
    runner = Runner(len(feature_keys))

    print('Training : ')
    for epoch in range(NUM_EPOCH):
        train_loss, train_acc, train_result = runner.run(train_loader, mode='train')
        valid_loss, valid_acc, valid_result = runner.run(valid_loader, mode='eval')
        print("[Epoch %d/%d] [Train Loss: %.4f] [Train Acc: %.4f%%] [Valid Loss: %.4f] [Valid Acc: %.4f%%]" %
              (epoch + 1, NUM_EPOCH, train_loss, train_acc, valid_loss, valid_acc))
        
    _, test_acc, test_result = runner.run(test_loader, mode='test')
    print("Training Finished")
    print("Training Accuracy: %.4f%%" % train_acc)
    utils.print_total_result(train_result)

    print("Validation Accuracy: %.4f%%" % valid_acc)
    utils.print_total_result(valid_result)

    print("Test Accuracy: %.4f%%" % test_acc)
    utils.print_total_result(test_result)

if __name__ == '__main__':
    main()
