import math
import os

from ml_sl.knn.knn_0 import KNN
from ml_sl.ml_data_wrapper import split_labeled_dataset_list
from ml_sl.ml_critrions import cal_accuracy, cal_kappa, cal_accuracy_on_2, cal_accuracy_on_3
from utils.file_utils.dataset_reader_pack.ml_dataset_reader import get_TV_T_dataset, get_T_V_T_dataset
from utils.file_utils.filename_utils import get_date_prefix

"""
本模块用于产生EA-Revise时要求的结果
	1- 网格搜索时，KNN结合最佳的五组超参数，分别计算【训练=tr】 和 【测试=va】上的 Accuracy 和 Kappa
	2- KNN结合最佳的超参数，分别计算【训练=tr+va】 和 【测试=te】上的 Accuracy 和 Kappa
"""

label_list = [2, 4, 5, 6, 7, 8, 9]

# Import dataset (Training, validation, Test)
ml_dataset_pickle_file_path = '../../datasets/ml_datasets/normed'
tr_dataset, va_dataset, te_dataset = get_T_V_T_dataset(file_path=ml_dataset_pickle_file_path)
tr_va_dataset, test_dataset = get_TV_T_dataset(file_path=ml_dataset_pickle_file_path)

tr_label_list, tr_data_list = split_labeled_dataset_list(tr_dataset)
va_label_list, va_data_list = split_labeled_dataset_list(va_dataset)
# print('tr data number', len(tr_label_list))
# print('va data number', len(va_label_list))
# print('te data number', len(te_dataset))
tr_va_label_list, tr_va_data_list = split_labeled_dataset_list(tr_va_dataset)
te_label_list, te_data_list = split_labeled_dataset_list(te_dataset)

# 1- 网格搜索时，KNN结合最佳的五组超参数，分别计算【训练=tr】 和 【测试=va】上的 Accuracy 和 Kappa
def knnGridSearchRes(outputFlag=True):
	top5_hyperPara_list = [['se_d', 6], ['manha_d', 14], ['e_d', 14], ['se_d', 5], ['jsd_d', 2]]
	
	for hyperPara in top5_hyperPara_list:
		distance_mode, k = hyperPara
		# Grid search training results (trained on tr-data, tested on tr-data)
		
		# knnTrRes = KNN(K = k,
		knnGSRes = KNN(K = k,
					   unlabeled_dataset_list = tr_data_list,
					   labeled_dataset_list = tr_dataset,
					   distance_mode = distance_mode,
					   label_list = label_list)
		# tr_sample_label_prob_dict_list = knnTrRes.classify()
		tr_sample_label_prob_dict_list = knnGSRes.classify()
		tr_acc = cal_accuracy(tr_sample_label_prob_dict_list, tr_label_list)
		tr_kappa = cal_kappa(tr_sample_label_prob_dict_list, tr_label_list)
		print('GS-Tr-K={0}-D={1}: Acc={2}, Kappa={3}'.format(k, distance_mode, tr_acc, tr_kappa))
		
		# Output the results into a txt
		if outputFlag:
			knnTrResFn = get_date_prefix() + 'knnGsTrResK={0}D={1}.txt'.format(k, distance_mode)
			knnTrResHeader = 'True_Label,Predict_Label\n'
			knnTrResF = open(knnTrResFn, 'a+')
			knnTrResF.write(knnTrResHeader)
			
			for true_label, label_prob_dict in zip(tr_label_list, tr_sample_label_prob_dict_list):
				max_prob_k_v_pair = max(label_prob_dict.items(), key=lambda k_v_pair: k_v_pair[1])
				pre_label = max_prob_k_v_pair[0]
				line_str = str(true_label) + ',' + str(pre_label) + '\n'
				knnTrResF.write(line_str)
			
			knnTrResF.close()

		# Grid search validation results (trained on tr-data, tested on va-data)
		knnGSRes.unlabeled_dataset_list = va_data_list
		va_sample_label_prob_dict_list = knnGSRes.classify()
		va_acc = cal_accuracy(va_sample_label_prob_dict_list, va_label_list)
		va_kappa = cal_kappa(va_sample_label_prob_dict_list, va_label_list)
		print('GS-Va-K={0}-D={1}: Acc={2}, Kappa={3}'.format(k, distance_mode, va_acc, va_kappa))
		
		# Output the results into a txt
		if outputFlag:
			knnVaResFn = get_date_prefix() + 'knnGsVaResK={0}D={1}.txt'.format(k, distance_mode)
			knnVaResF = open(knnVaResFn, 'a+')
			knnVaResF.write(knnTrResHeader)
			
			for true_label, label_prob_dict in zip(va_label_list, va_sample_label_prob_dict_list):
				max_prob_k_v_pair = max(label_prob_dict.items(), key=lambda k_v_pair: k_v_pair[1])
				pre_label = max_prob_k_v_pair[0]
				line_str = str(true_label) + ',' + str(pre_label) + '\n'
				knnVaResF.write(line_str)
			
			knnVaResF.close()
# knnGridSearchRes(outputFlag=True)
knnGridSearchRes(outputFlag=False)

"""
1- 计算网格搜索时，KNN 分别 结合最佳的五组超参数 后，的 Accuracy 和 Kappa
	GS-Tr-K=6-D=se_d: Acc=0.6509009009009009, Kappa=0.5512782896152417, AK=1.2021791905161425
	GS-Va-K=6-D=se_d: Acc=0.5531914893617021, Kappa=0.4208596156667156, AK=0.9740511050284177
	GS-Tr-K=14-D=manha_d: Acc=0.5675675675675675, Kappa=0.4365705674743228, AK=1.0041381350418903
	GS-Va-K=14-D=manha_d: Acc=0.5425531914893617, Kappa=0.3952722920406941, AK=0.9378254835300557
	GS-Tr-K=14-D=e_d: Acc=0.5630630630630631, Kappa=0.42947601290262766, AK=0.9925390759656907
	GS-Va-K=14-D=e_d: Acc=0.5425531914893617, Kappa=0.3934573829531812, AK=0.9360105744425429
	GS-Tr-K=5-D=se_d: Acc=0.6509009009009009, Kappa=0.5547100956965663, AK=1.2056109965974673
	GS-Va-K=5-D=se_d: Acc=0.5319148936170213, Kappa=0.3968207670993145, AK=0.9287356607163357
	GS-Tr-K=2-D=jsd_d: Acc=0.7027027027027027, Kappa=0.6187973592637159, AK=1.3215000619664186
	GS-Va-K=2-D=jsd_d: Acc=0.5319148936170213, Kappa=0.3897904986721747, AK=0.921705392289196
"""