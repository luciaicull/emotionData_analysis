import torch
import numpy as np
import pandas as pd
import time

from constants import NUM_EPOCH, FEATURE_KEYS, LEARNING_RATE
import data_manager
import models

class Runner(object):
    def __init__(self):
        self.model = models.SimpleClassifier()
        self.model = self.model.double()
        
        #self.criterion = torch.nn.MultiMarginLoss()
        self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE)

        self.device = torch.device("cpu")

    def get_result(self, pred_list, answ_list):
        # pred_list : list of pred (= [0.7, 0.1, 0.05, 0.1, 0.05])
        # answ_list : list of ans (= [1, 0, 0, 0, 0])
        result = [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]
        for pred, ans in zip(pred_list, answ_list):
            p = pred.index(max(pred))
            #a = ans.index(max(ans))
            a = ans[0]
            result[a][p] += 1
        
        ratio_result = [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]
        for i, emotion in enumerate(result):
            for j, pred_num in enumerate(emotion):
                ratio_result[i][j] = pred_num/sum(emotion)

        total_result = self._get_total_result(result, ratio_result)
        total_accuracy = self._get_total_accuracy(result)

        return total_result, total_accuracy
    
    def _get_total_accuracy(self, result):
        total_num = 0
        correct_num = 0
        for i, res in enumerate(result):
            total_num += sum(res)
            correct_num += res[i]
        return (correct_num / total_num) * 100
    
    def _get_total_result(self, result, ratio_result):
        total_result = dict()
        # 1. Single emotion -> Single emotion accuracy
        total_result['single_to_single'] = ratio_result

        # 2. Single emotion -> Arousal accuracy
        e2_to_HA = ratio_result[1][3] + ratio_result[1][4]
        e2_to_LA = ratio_result[1][1] + ratio_result[1][2]
        e3_to_HA = ratio_result[2][3] + ratio_result[2][4]
        e3_to_LA = ratio_result[2][2] + ratio_result[2][1]
        e4_to_HA = ratio_result[3][3] + ratio_result[3][4]
        e4_to_LA = ratio_result[3][1] + ratio_result[3][2]
        e5_to_HA = ratio_result[4][4] + ratio_result[4][3]
        e5_to_LA = ratio_result[4][1] + ratio_result[4][2]
        total_result['single_to_arousal'] = [[e2_to_HA, e2_to_LA],
                                            [e3_to_HA, e3_to_LA],
                                            [e4_to_HA, e4_to_LA],
                                            [e5_to_HA, e5_to_LA]]

        # 3. Single emotion -> Valence accuracy
        e2_to_PV = ratio_result[1][1] + ratio_result[1][3]
        e2_to_NV = ratio_result[1][2] + ratio_result[1][4]
        e3_to_PV = ratio_result[2][1] + ratio_result[2][3]
        e3_to_NV = ratio_result[2][2] + ratio_result[2][4]
        e4_to_PV = ratio_result[3][3] + ratio_result[3][1]
        e4_to_NV = ratio_result[3][2] + ratio_result[3][4]
        e5_to_PV = ratio_result[4][1] + ratio_result[4][3]
        e5_to_NV = ratio_result[4][4] + ratio_result[4][2]
        total_result['single_to_valence'] = [[e2_to_PV, e2_to_NV],
                                            [e3_to_PV, e3_to_NV],
                                            [e4_to_PV, e4_to_NV],
                                            [e5_to_PV, e5_to_NV]]

        # 4. Arousal -> Arousal accuracy
        HA_to_HA = (result[3][3]+result[3][4]+result[4][3] +
                    result[4][4]) / (sum(result[3]) + sum(result[4]))
        HA_to_LA = (result[3][1]+result[3][2]+result[4][1] +
                    result[4][2]) / (sum(result[3]) + sum(result[4]))
        LA_to_LA = (result[1][1]+result[1][2]+result[2][1] +
                    result[2][2]) / (sum(result[1])+sum(result[2]))
        LA_to_HA = (result[1][3]+result[1][4]+result[2][3] +
                    result[2][4]) / (sum(result[1])+sum(result[2]))
        total_result['arousal_to_arousal'] = [[HA_to_HA, HA_to_LA],
                                            [LA_to_HA, LA_to_LA]]

        # 5. Valence -> Valence accuracy
        PV_to_PV = (result[1][1]+result[1][3]+result[3][1] +
                    result[3][3]) / (sum(result[1]) + sum(result[3]))
        PV_to_NV = (result[1][2]+result[1][4]+result[3][2] +
                    result[3][4]) / (sum(result[1]) + sum(result[3]))
        NV_to_NV = (result[2][2]+result[2][4]+result[4][2] +
                    result[4][4]) / (sum(result[2]) + sum(result[4]))
        NV_to_PV = (result[2][1]+result[2][3]+result[4][1] +
                    result[4][3]) / (sum(result[2]) + sum(result[4]))
        total_result['valence_to_valence'] = [[PV_to_PV, PV_to_NV],
                                            [NV_to_PV, NV_to_NV]]

        return total_result

    def print_total_result(self, total_result):
        for key in total_result.keys():
            print(key)
            df = pd.DataFrame(total_result[key])
            #display(df)
            print(df)
            


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
        total_result, total_accuracy = self.get_result(prediction_list, answer_list)

        return epoch_loss, total_accuracy, total_result  

def main():
    '''
    cur = time.localtime()
    cur_time = "%04d-%02d-%02d_%02d:%02d" % (cur.tm_year, cur.tm_mon, cur.tm_mday, cur.tm_hour, cur.tm_min)
    filename = './_experiments/' + cur_time
    f = open(filename + '.txt', 'w')
    f.write('feature keys\n')
    f.write(FEATURE_KEYS)
    f.write('\n\n')

    f.write('model : 2layer, input-input-output\n\n')

    '''
    #202008250137
    torch.manual_seed(1234)

    train_loader, valid_loader, test_loader = data_manager.get_dataloader()
    runner = Runner()

    train_loss_list = []
    valid_loss_list = []
    print('Training : ')
    for epoch in range(NUM_EPOCH):
        train_loss, train_acc, train_result = runner.run(train_loader, mode='train')
        valid_loss, valid_acc, valid_result = runner.run(valid_loader, mode='eval')
        print("[Epoch %d/%d] [Train Loss: %.4f] [Train Acc: %.4f%%] [Valid Loss: %.4f] [Valid Acc: %.4f%%]" %
              (epoch + 1, NUM_EPOCH, train_loss, train_acc, valid_loss, valid_acc))
        #f.write("[Epoch %d/%d] [Train Loss: %.4f] [Train Acc: %.4f%%] [Valid Loss: %.4f] [Valid Acc: %.4f%%]\n" %
        #        (epoch + 1, NUM_EPOCH, train_loss, train_acc, valid_loss, valid_acc))
    
    test_loss, test_acc, test_result = runner.run(test_loader, mode='test')
    print("Training Finished")
    print("Training Accuracy: %.4f%%" % train_acc)
    runner.print_total_result(train_result)

    print("Validation Accuracy: %.4f%%" % valid_acc)
    runner.print_total_result(valid_result)

    print("Test Accuracy: %.4f%%" % test_acc)
    runner.print_total_result(test_result)

if __name__ == '__main__':
    main()
