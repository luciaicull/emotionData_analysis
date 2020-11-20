import torch
import torch.nn as nn
import numpy as np
import csv

from .arg_parser import get_parser
from . import data_manager, models, utils
from .constants import FEATURE_KEYS


class Runner(object):
    def __init__(self, input_size, learning_rate, pooling):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pooling =pooling
        self.model = models.AdaptivePoolingClassifier(self.device, input_size, pooling=pooling).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def run(self, dataloader, mode='train', save_result=False, save_name=None):
        if mode is 'test' and save_result:
            csv_file = open('/home/yoojin/data/emotionDataset/final/save/'+save_name+'_'+mode+'.csv', 'w')
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(['name', 'measure range','result', 'answer'])

        self.model.train() if mode is 'train' else self.model.eval()

        epoch_loss = 0
        prediction_list = []
        answer_list = []
        name_list = []

        self.model = self.model.float()
        frag_results = dict()
        for _, (x_list, y, name, measure_list) in enumerate(dataloader):
            x_list = x_list.to(self.device, dtype=torch.float)
            prediction, predicted_x_list = self.model(x_list)

            y = y.to(self.device)

            loss = self.criterion(prediction, y)

            if mode is 'train':
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_loss += prediction.size(0) * loss.item()

            sm = torch.nn.Softmax()
            pred_prob = sm(prediction)
            prediction_list.append(pred_prob.tolist()[0])
            answer_list.append(y.tolist()[0])
            name_list.append(name)

            if mode is 'test' and save_result:
                if predicted_x_list.size()[1] != 1:
                    predicted_x_list = predicted_x_list.squeeze()
                else:
                    predicted_x_list = predicted_x_list.squeeze(dim=1)
                for measure_range, pred_x_list in zip(measure_list, predicted_x_list):
                    meas = [t.item() for t in measure_range]
                    res = sm(pred_x_list).tolist()
                    writer.writerow([name, meas, res, y.item()])
                #frag_results[name] = {'measures':[[t.item() for t in positions] for positions in measure_list]}
                #frag_results[name]['frag_results'] = sm(predicted_x_list.squeeze()).tolist()

        epoch_loss = epoch_loss / len(dataloader.dataset)

        if mode is not 'test':
            total_result, total_accuracy = utils.get_result(prediction_list, answer_list)
        else:
            if save_result:
                csv_file.close()

            # print results
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
            for pred, ans, name in zip(prediction_list, answer_list, name_list):
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


        if mode is 'train' and self.pooling == "auto":
            print(self.model.alpha)
        return epoch_loss, total_accuracy, total_result


def main():
    seed = 0
    print(seed)
    torch.manual_seed(seed)

    p = get_parser()
    args = p.parse_args()
    feature_keys = FEATURE_KEYS

    train_loader, valid_loader, test_loader = data_manager.get_dataloader(args.path, args.data_name, feature_keys, args.batch_size)
    #train_loader, test_loader = data_manager.get_dataloader(args.path, args.data_name, feature_keys, args.batch_size)

    runner = Runner(len(feature_keys), args.learning_rate, args.pooling)

    print('Training : ')
    for epoch in range(args.num_epoch):
        train_loss, train_acc, train_result = runner.run(train_loader, mode='train')
        valid_loss, valid_acc, valid_result = runner.run(valid_loader, mode='eval')
        print("[Epoch %d/%d] [Train Loss: %.4f] [Train Acc: %.4f%%] [Valid Loss: %.4f] [Valid Acc: %.4f%%]" %
              (epoch + 1, args.num_epoch, train_loss, train_acc, valid_loss, valid_acc))

    _, test_acc, test_result = runner.run(test_loader, mode='test', save_result=True, save_name=args.data_name[:-len('.dat')])
    print("Training Finished")
    print("Training Accuracy: %.4f%%" % train_acc)
    utils.print_total_result(train_result)

    print("Validation Accuracy: %.4f%%" % valid_acc)
    utils.print_total_result(valid_result)

    print("Test Accuracy: %.4f%%" % test_acc)
    utils.print_total_result(test_result)


if __name__ == "__main__":
    main()
