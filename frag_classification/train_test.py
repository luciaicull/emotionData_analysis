import torch
import numpy as np
from .arg_parser import get_parser
from . import data_manager, models, utils
from .constants import TEST_FEATURE_KEYS, LEARNING_RATE, NUM_EPOCH, FEATURE_KEYS


class Runner(object):
    def __init__(self, input_size):
        self.model = models.SimpleClassifier(input_size)
        self.model = self.model.double()

        #self.criterion = torch.nn.MultiMarginLoss()
        self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
    
    def run(self, dataloader, mode='train', test_with_split=True):
        # test_with_split => False: valid/test를 total로 // True: valid/test를 fragment로

        self.model.train() if mode is 'train' else self.model.eval()

        epoch_loss = 0
        prediction_list = []
        answer_list = []
        for _, (x, y) in enumerate(dataloader):
            if mode != 'train' and test_with_split:
                frag_prediction_list = torch.as_tensor([], dtype=torch.double).to(self.device)
                for frag in x:
                    frag_x = torch.as_tensor([], dtype=torch.double).to(self.device)
                    for f in frag:
                        f = torch.unsqueeze(f, dim=0).to(self.device, dtype=torch.double)
                        frag_x = torch.cat((frag_x, f))
                    #frag_x = torch.unsqueeze(frag_x, dim=0)
                    prediction = self.model(frag_x)
                    prediction = torch.unsqueeze(prediction, dim=0)
                    frag_prediction_list = torch.cat((frag_prediction_list, prediction))
                prediction = torch.mean(frag_prediction_list, axis=0)
            else:
                x = x.to(self.device)
                prediction = self.model(x)
            
            y = y.to(self.device, dtype=torch.long)

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
    torch.manual_seed(1234)

    p = get_parser()
    args = p.parse_args()
    #feature_keys = TEST_FEATURE_KEYS
    feature_keys = FEATURE_KEYS
    test_with_split = False  # True: valid/test를 fragment로 // False: valid/test를 total로

    # test with split
    # TODO

    # test without split
    train_loader, valid_loader, test_loader = data_manager.get_dataloader(args.path, args.frag_data_name, args.total_data_name, feature_keys, test_with_split=test_with_split)
    runner = Runner(len(feature_keys))

    print('Training : ')
    for epoch in range(NUM_EPOCH):
        train_loss, train_acc, train_result = runner.run(train_loader, mode='train', test_with_split=test_with_split)
        valid_loss, valid_acc, valid_result = runner.run(valid_loader, mode='eval', test_with_split=test_with_split)
        print("[Epoch %d/%d] [Train Loss: %.4f] [Train Acc: %.4f%%] [Valid Loss: %.4f] [Valid Acc: %.4f%%]" %
              (epoch + 1, NUM_EPOCH, train_loss, train_acc, valid_loss, valid_acc))
        
    _, test_acc, test_result = runner.run(test_loader, mode='test', test_with_split=test_with_split)
    print("Training Finished")
    print("Training Accuracy: %.4f%%" % train_acc)
    utils.print_total_result(train_result)

    print("Validation Accuracy: %.4f%%" % valid_acc)
    utils.print_total_result(valid_result)

    print("Test Accuracy: %.4f%%" % test_acc)
    utils.print_total_result(test_result)


if __name__ == "__main__":
    main()
