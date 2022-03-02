def cal_accuracy(sample_label_prob_dict_list, test_label_list):
    """
    :param
        sample_label_prob_dict_list:
            [
                {1: 0.2, 2:0.15, 3:0.2, ..., 9:0.1}
                {1: 0.2, 2:0.15, 3:0.2, ..., 9:0.1}
                ...
            ]
        test_label_list:
            [1,2,5,6,8, ...]
    :return:
        acc
            accuracy
    """
    test_correct_count = 0
    for label_prob_dict, label in zip(sample_label_prob_dict_list, test_label_list):
        max_prob_k_v_pair = max(label_prob_dict.items(), key=lambda k_v_pair: k_v_pair[1])
        key = max_prob_k_v_pair[0]
        if key == label:
            test_correct_count += 1
    acc = test_correct_count / len(test_label_list)
    return acc

def cal_accuracy_on_2(sample_label_prob_dict_list, test_label_list):
    """
    :param
        sample_label_prob_dict_list:
            [
                {1: 0.2, 2:0.15, 3:0.2, ..., 9:0.1}
                {1: 0.2, 2:0.15, 3:0.2, ..., 9:0.1}
                ...
            ]
        test_label_list:
            [1,2,5,6,8, ...]
    :return:
        accuracy of the first two predictions
    """
    test_correct_count = 0
    for label_prob_dict, label in zip(sample_label_prob_dict_list, test_label_list):
        first_2_max_prob_k_v_pair = sorted(label_prob_dict.items(), key=lambda k_v_pair: k_v_pair[1], reverse=True)[:2]
        key1 = first_2_max_prob_k_v_pair[0][0]
        key2 = first_2_max_prob_k_v_pair[1][0]

        if (key1 == label) or (key2 == label):
            test_correct_count += 1

    acc_on_2 = test_correct_count / len(test_label_list)
    return acc_on_2

def cal_accuracy_on_3(sample_label_prob_dict_list, test_label_list):
    """
    :param
        sample_label_prob_dict_list:
            [
                {1: 0.2, 2:0.15, 3:0.2, ..., 9:0.1}
                {1: 0.2, 2:0.15, 3:0.2, ..., 9:0.1}
                ...
            ]
        test_label_list:
            [1,2,5,6,8, ...]
    :return:
        accuracy of the first three predictions
    """
    test_correct_count = 0
    for label_prob_dict, label in zip(sample_label_prob_dict_list, test_label_list):
        first_3_max_prob_k_v_pair = sorted(label_prob_dict.items(), key=lambda k_v_pair: k_v_pair[1], reverse=True)[:3]
        key1 = first_3_max_prob_k_v_pair[0][0]
        key2 = first_3_max_prob_k_v_pair[1][0]
        key3 = first_3_max_prob_k_v_pair[2][0]

        if (key1 == label) or (key2 == label) or (key3 == label):
            test_correct_count += 1

    acc_on_3 = test_correct_count / len(test_label_list)
    return acc_on_3

def cal_kappa(sample_label_prob_dict_list, test_label_list):
    """
    Reference:
        机器学习中多分类模型的评估方法之--kappa系数
        https://blog.csdn.net/wang7807564/article/details/80252362
    :param
        sample_label_prob_dict_list:
            [
                {1: 0.2, 2:0.15, 3:0.2, ..., 9:0.1}
                {1: 0.2, 2:0.15, 3:0.2, ..., 9:0.1}
                ...
            ]
        test_label_list:
            [1,2,5,6,8, ...]
    :return:
        acc
            accuracy
    """
    label_list = list(set(test_label_list))
    label_dict_dict = {}
    for label in label_list:
        label_dict = {}
        for i in label_list:
            label_dict[i] = 0
        label_dict_dict[label] = label_dict

    test_len = len(test_label_list)
    for sample_label_prob_dict, label in zip(sample_label_prob_dict_list, test_label_list):
        max_prob_k_v_pair = max(sample_label_prob_dict.items(), key=lambda k_v_pair: k_v_pair[1])
        key = max_prob_k_v_pair[0]
        label_dict_dict[label][key] += 1

    # Calculate P0
    p0 = sum([label_dict_dict[key][key] for key in label_dict_dict]) / test_len

    # Calculate Pe
    pe = 0.0
    for label in label_list:
        label_real_count = sum(label_dict_dict[label].values())
        label_prediction_count = sum([label_item[1][label] for label_item in label_dict_dict.items()])
        pe += label_real_count * label_prediction_count / (test_len * test_len)

    k = (p0 - pe) / (1 - pe)
    return k