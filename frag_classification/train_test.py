import torch
import torch.nn as nn
import numpy as np
import csv

from .arg_parser import get_parser
from . import data_manager, models, utils
from .constants import LEARNING_RATE, FEATURE_KEYS


class Runner(object):
    def __init__(self, input_size):
        self.model = models.SimpleClassifier(input_size)
        self.model = self.model.double()

        self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
    
    def run(self, dataloader, mode='train', save_result=False, save_name=None):
        if save_result:
            csv_file = open('/home/yoojin/data/emotionDataset/final/save/'+save_name+'_'+mode+'.csv', 'w')
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(['name', 'measure range','result', 'answer'])

        self.model.train() if mode is 'train' else self.model.eval()

        epoch_loss = 0
        prediction_list = []
        answer_list = []
        names = []
        for _, (x, y, name, measure) in enumerate(dataloader):
            if mode != 'train':
                frag_prediction_list = torch.as_tensor([], dtype=torch.double).to(self.device)
                for frag, measure_range in zip(x, measure):
                    frag_x = torch.as_tensor([], dtype=torch.double).to(self.device)
                    for f in frag:
                        f = torch.unsqueeze(f, dim=0).to(self.device, dtype=torch.double)
                        frag_x = torch.cat((frag_x, f))
                    prediction = self.model(frag_x)
                    prediction = torch.unsqueeze(prediction, dim=0)
                    frag_prediction_list = torch.cat((frag_prediction_list, prediction))
                    
                    # test fragment softmax result
                    m = nn.Softmax(dim=1)
                    softmax_result = m(prediction)
                    if save_result:
                        writer.writerow([name, [t.item() for t in measure_range], softmax_result.tolist(), y.item()])

                prediction = torch.mean(frag_prediction_list, axis=0)
                if mode is 'test':
                    names.append(name)
            else:
                x = x.to(self.device)
                prediction = self.model(x)
                if mode is 'test':
                    names.append(name)
            
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
        if mode is not 'test':
            total_result, total_accuracy = utils.get_result(prediction_list, answer_list)
        else:
            total_result, total_accuracy = utils.get_result(prediction_list, answer_list, True)
            COMPOSER_PERIOD_DICT = {'Bach': 'Baroque', 'Badarzewska-Baranowska': 'Romantic', 'Bartok': 'Modern', 'Beethoven': 'Classical',
                                    'Brahms': 'Romantic', 'Chopin': 'Romantic', 'Clementi': 'Classical', 'Debussy': 'Romantic',
                                    'Hanon': 'Classical', 'Kuhlau': 'Classical', 'Liszt': 'Romantic', 'Mendelssohn': 'Romantic',
                                    'Messiaen': 'Modern', 'Mompou': 'Modern', 'Mozart': 'Classical', 'Prokofiev': 'Modern', 'Rachmaninoff': 'Romantic',
                                    'Scarlatti': 'Baroque', 'Schubert': 'Romantic', 'Schumann': 'Romantic', 'Scriabin': 'Romantic', 'Tchaikovsky': 'Romantic'
                                    }
            period_result = {'Baroque': [[], []],
                             'Classical': [[], []],
                             'Romantic': [[], []],
                             'Modern': [[], []]}
            composer_result = dict()
            for pred, ans, name in zip(prediction_list, answer_list, names):
                composer = name[0].split('.')[0]
                period = COMPOSER_PERIOD_DICT[composer]
                # period result
                period_result[period][0].append(pred)
                period_result[period][1].append(ans)
                # composer result
                if composer not in composer_result.keys():
                    composer_result[composer] = [[], []]
                composer_result[composer][0].append(pred)
                composer_result[composer][1].append(ans)

            for period in period_result.keys():
                period_total_result, period_total_accuracy = utils.get_result(
                    period_result[period][0], period_result[period][1])
                print(period)
                print(period_total_accuracy)
                utils.print_total_result(period_total_result)
            for composer in composer_result.keys():
                composer_total_result, composer_total_accuracy = utils.get_result(composer_result[composer][0], composer_result[composer][1])
                print(composer)
                print(composer_total_accuracy)
                utils.print_total_result(composer_total_result)

        if save_result:
            csv_file.close()

        return epoch_loss, total_accuracy, total_result 

def main():
    torch.manual_seed(1234)

    p = get_parser()
    args = p.parse_args()
    feature_keys = FEATURE_KEYS

    #train_loader, valid_loader, test_loader = data_manager.get_dataloader(args.path, args.frag_data_name, feature_keys)
    train_loader, test_loader = data_manager.get_dataloader(args.path, args.frag_data_name, feature_keys)
    runner = Runner(len(feature_keys))

    num_epoch = args.num_epoch
    print('Training : ')
    for epoch in range(num_epoch):
        train_loss, train_acc, train_result = runner.run(train_loader, mode='train')
        valid_loss, valid_acc, valid_result = runner.run(test_loader, mode='eval')
        print("[Epoch %d/%d] [Train Loss: %.4f] [Train Acc: %.4f%%] [Valid Loss: %.4f] [Valid Acc: %.4f%%]" %
              (epoch + 1, num_epoch, train_loss, train_acc, valid_loss, valid_acc))
    
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
