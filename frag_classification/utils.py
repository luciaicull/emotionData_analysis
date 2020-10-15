import pandas as pd

def get_result(pred_list, answ_list):
    # pred_list : list of pred (= [0.7, 0.1, 0.05, 0.1, 0.05])
    # answ_list : list of ans (= [1, 0, 0, 0, 0])
    result = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    for pred, ans in zip(pred_list, answ_list):
        p = pred.index(max(pred))
        a = ans[0]
        result[a][p] += 1

    ratio_result = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    for i, emotion in enumerate(result):
        for j, pred_num in enumerate(emotion):
            ratio_result[i][j] = pred_num/sum(emotion)

    total_result = get_total_result(result, ratio_result)
    total_accuracy = get_total_accuracy(result)

    return total_result, total_accuracy

def get_total_accuracy(result):
    total_num = 0
    correct_num = 0
    for i, res in enumerate(result):
        total_num += sum(res)
        correct_num += res[i]
    return (correct_num / total_num) * 100


def get_total_result(result, ratio_result):
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


def print_total_result(total_result):
    for key in total_result.keys():
        print(key)
        df = pd.DataFrame(total_result[key])
        #display(df)
        print(df)
