import os
import sys
import time
import math
from utils.file_utils.filename_utils import get_date_prefix

from utils.file_utils.dataset_reader_pack.ml_dataset_reader import get_TV_T_dataset, get_T_V_T_dataset
from ml_sl.ml_data_wrapper import split_labeled_dataset_list
from ml_sl.ml_critrions import cal_accuracy, cal_kappa, cal_accuracy_on_2, cal_accuracy_on_3
from ml_sl.logistic.lrc_0 import LRC

"""
本模块用于产生EA-Revise时要求的结果
	1- 网格搜索时，LRC结合最佳的五组超参数，分别计算【训练=tr】 和 【测试=va】上的 Accuracy 和 Kappa
	2- LRC结合最佳的超参数，分别计算【训练=tr+va】 和 【测试=te】上的 Accuracy 和 Kappa
"""

label_list = [2, 4, 5, 6, 7, 8, 9]
# Import dataset (Training, validation, Test)
ml_dataset_pickle_file_path = '../../datasets/ml_datasets/normed'
tr_dataset, va_dataset, te_dataset = get_T_V_T_dataset(file_path=ml_dataset_pickle_file_path)
tr_va_dataset, test_dataset = get_TV_T_dataset(file_path=ml_dataset_pickle_file_path)

tr_label_list, tr_data_list = split_labeled_dataset_list(tr_dataset)
va_label_list, va_data_list = split_labeled_dataset_list(va_dataset)
tr_va_label_list, tr_va_data_list = split_labeled_dataset_list(tr_va_dataset)
te_label_list, te_data_list = split_labeled_dataset_list(te_dataset)

def lrcGridSearchRes(classificationStrategy='OvO'):
	# top5HyperParaList=[[linear Learning rate-alpha = 10^3, Iteration time = 5k], [],[]]
	top5HyperParaList = [[1e3, 5e3], [1e2, 5e3], [10, 5e3], [1e2, 7e3], [1e3, 9e3]]
	
	# 因为每组参数都被重复训练了10次，要从已经训练好的模型文件中，将每组参数对应的10个模型找出来
	# 所有GS-LRC-OvR的模型文件都存放在：dpfc_src\ml_sl\logistic\ovr_res\linear\models
	# modelFp = 'ovr_res/linear/models/'
	# modelFp = 'ovr_res/linear/models/1'
	modelFp = 'ovo_res/linear/models/'
	fnList = os.listdir(modelFp)
	
	for hyperPara in top5HyperParaList:
		# ------------------------------- 将每组参数对应的10个模型找出来 -------------------------------
		targetFnList = []
		for fn in fnList:
			if classificationStrategy == 'OvO':
				"""
				OvO
				所有的模型文件 分 两次 训练完成，两次训练模型的命名方式不同，
					第一次训练是在 2020-4-06 ~ 2020-4-11
						2020_04_06_lrc_linear_alpha_iter_1000_alpha_init_1.0_classifer_dict_pickle_0.file
						2020_04_11_lrc_linear_alpha_iter_9000_alpha_init_1.0_classifer_dict_pickle_4.file
					第二次训练是在 2020-04-20 ~ 2020-04-22
						2020_04_20_lrc_ovo_linear_alpha_iter_3000_alpha_init_100_classifer_dict_pickle_4.file
						2020_04_22_lrc_ovo_linear_alpha_iter_15000_alpha_init_1000_classifer_dict_pickle_9.file
				"""
				fnStrList = fn.split('_')
				
				# 根据文件 day 判断
				dayStr = fnStrList[2]
				alphaFloat, iterFloat = None, None
				if dayStr in ['06','07','08','09','10','11']:
					alphaFloat = float(fnStrList[10])
					iterFloat = float(fnStrList[7])
				elif dayStr in ['20', '21', '22']:
					alphaFloat = float(fnStrList[11])
					iterFloat = float(fnStrList[8])
				if (hyperPara[0] == alphaFloat) and (hyperPara[1] == iterFloat):
					targetFnList.append(fn)
			elif classificationStrategy == 'OvR':
				"""
				OvR
				所有的模型文件 分 两次 训练完成，两次训练模型的命名方式不同，
					第一次训练是在 2020-4-18 ~ 2020-4-20
						2020_04_18_lrc_ovr_linear_alpha_iter_1000_alpha_init=0.0001_classifer_dict_pickle_7.file
						2020_04_18_lrc_ovr_linear_alpha_iter_1000_alpha_init=0.01_classifer_dict_pickle_8.file
					第二次训练是在 2020-05-04 ~ 2020-05-07
						2020_05_04_lrc_ovr_linear_iter=11000_alpha_init=0.0001_classifer_dict_pickle_8.file
						2020_05_04_lrc_ovr_linear_iter=13000_alpha_init=0.01_classifer_dict_pickle_5.file
				"""
				fnStrList = fn.split('_')
				
				# 根据文件月份判断
				monthInt = int(fnStrList[1])
				
				if monthInt == 4:
					# fnStrList[8] == '11000', --> int
					iterInt = int(fnStrList[8])
					
					# fnStrList[10] == 'init=0.0001', --> int(0.01) == 0, int(100) == 100
					try:
						alphaInt = int(fnStrList[10].split('=')[1])
					except ValueError as e:
						alphaInt = 0
						print(e)
				elif monthInt == 5:
					# fnStrList[6] == 'iter=11000', --> int
					iterInt = int(fnStrList[6].split('=')[1])
					
					# fnStrList[8] == 'init=0.0001', --> int(0.01) == 0, int(100) == 100
					try:
						alphaInt = int(fnStrList[8].split('=')[1])
					except ValueError as e:
						alphaInt = 0
						print(e)
				
				if (int(hyperPara[0]) == alphaInt) and (int(hyperPara[1]) == iterInt):
					targetFnList.append(fn)
		# ------------------------------- 将每组参数对应的10个模型找出来 -------------------------------
		
		# ------------------------------- 依次加载 10个模型 分别进行实验 -------------------------------
		lrcGSTrAccList = []
		lrcGSTrKappaList = []
		lrcGSVaAccList = []
		lrcGSVaKappaList = []
		for targetFn in targetFnList:
			modelFn = os.path.join(modelFp, targetFn)
			# create an basically empty LRC, just need its classify_ovr function
			# --------------------------------- GS get results on tr ---------------------------------
			lrcGSTr = LRC(alpha=0, max_iter=0,
						  unlabeled_dataset_list=tr_data_list,
						  labeled_dataset_list=None,
						  label_list=label_list)
			if classificationStrategy == 'OvO':
				lrcGSTrSample_label_prob_dict_list = lrcGSTr.classify_ovo(lrc_classifer_dict_pickle_filename=modelFn)
			elif classificationStrategy == 'OvR':
				lrcGSTrSample_label_prob_dict_list = lrcGSTr.classify_ovr(lrc_ovr_classifer_dict_pickle_filename=modelFn)
			lrcGSTrAcc = cal_accuracy(lrcGSTrSample_label_prob_dict_list, tr_label_list)
			lrcGSTrAccList.append(lrcGSTrAcc)
			lrcGSTrKappa = cal_kappa(lrcGSTrSample_label_prob_dict_list, tr_label_list)
			lrcGSTrKappaList.append(lrcGSTrKappa)
			# --------------------------------- GS get results on tr ---------------------------------
			
			# --------------------------------- GS get results on va ---------------------------------
			lrcGSVa = LRC(alpha=0, max_iter=0,
						  unlabeled_dataset_list=va_data_list,
						  labeled_dataset_list=None,
						  label_list=label_list)
			if classificationStrategy == 'OvO':
				lrcGSVaSample_label_prob_dict_list = lrcGSVa.classify_ovo(lrc_classifer_dict_pickle_filename=modelFn)
			elif classificationStrategy == 'OvR':
				lrcGSVaSample_label_prob_dict_list = lrcGSVa.classify_ovr(lrc_ovr_classifer_dict_pickle_filename=modelFn)
			lrcGSVaAcc = cal_accuracy(lrcGSVaSample_label_prob_dict_list, va_label_list)
			lrcGSVaAccList.append(lrcGSVaAcc)
			lrcGSVaKappa = cal_kappa(lrcGSVaSample_label_prob_dict_list, va_label_list)
			lrcGSVaKappaList.append(lrcGSVaKappa)
			# --------------------------------- GS get results on va ---------------------------------
		
			# print result
			print('GS:Alpha={0},Iter={1}:Tr-ACC={2},Tr-Kappa={3},Va-Acc={4},Va-Kappa={5}'.format(hyperPara[0],
																								 hyperPara[1],
																								 lrcGSTrAcc,lrcGSTrKappa,
																								 lrcGSVaAcc,lrcGSVaKappa))
		print('GS:Alpha={0},Iter={1}:Tr-ACC-Avg={2},Tr-Kappa-Avg={3},Va-Acc-Avg={4},Va-Kappa-Avg={5}'.format(
			hyperPara[0], hyperPara[1],
			sum(lrcGSTrAccList) / len(lrcGSTrAccList), sum(lrcGSTrKappaList) / len(lrcGSTrKappaList),
			sum(lrcGSVaAccList) / len(lrcGSVaAccList), sum(lrcGSVaKappaList) / len(lrcGSVaKappaList)
		))
		# ------------------------------- 依次加载 10个模型 分别进行实验 -------------------------------
# lrcGridSearchRes()
"""
GS-LRC-OvO
GS:Alpha=1000.0,Iter=5000.0:Tr-ACC=0.8153153153153153,Tr-Kappa=0.7713582355748702.Va-Acc=0.5,Va-Kappa=0.390200138026225
GS:Alpha=1000.0,Iter=5000.0:Tr-ACC=0.8198198198198198,Tr-Kappa=0.7767105031557241.Va-Acc=0.5,Va-Kappa=0.390200138026225
GS:Alpha=1000.0,Iter=5000.0:Tr-ACC=0.8153153153153153,Tr-Kappa=0.7713323870416661.Va-Acc=0.5,Va-Kappa=0.390200138026225
GS:Alpha=1000.0,Iter=5000.0:Tr-ACC=0.8153153153153153,Tr-Kappa=0.7713122785859652.Va-Acc=0.5,Va-Kappa=0.390200138026225
GS:Alpha=1000.0,Iter=5000.0:Tr-ACC=0.8175675675675675,Tr-Kappa=0.7740330242026691.Va-Acc=0.5,Va-Kappa=0.390200138026225
GS:Alpha=1000.0,Iter=5000.0:Tr-ACC=0.8198198198198198,Tr-Kappa=0.7769152504050947.Va-Acc=0.5,Va-Kappa=0.390200138026225
GS:Alpha=1000.0,Iter=5000.0:Tr-ACC=0.8153153153153153,Tr-Kappa=0.7713381316652221.Va-Acc=0.5,Va-Kappa=0.39062068965517244
GS:Alpha=1000.0,Iter=5000.0:Tr-ACC=0.8130630630630631,Tr-Kappa=0.7686396624896411.Va-Acc=0.5,Va-Kappa=0.390200138026225
GS:Alpha=1000.0,Iter=5000.0:Tr-ACC=0.8130630630630631,Tr-Kappa=0.7684535927015005.Va-Acc=0.5,Va-Kappa=0.39062068965517244
GS:Alpha=1000.0,Iter=5000.0:Tr-ACC=0.8085585585585585,Tr-Kappa=0.7630914866103374.Va-Acc=0.5,Va-Kappa=0.390200138026225
GS:Alpha=1000.0,Iter=5000.0:
	Tr-ACC-Avg=0.8153153153153152,	Tr-Kappa-Avg=0.7713184552432689,	AK=1.586633770558584
	Va-Acc-Avg=0.5,					Va-Kappa-Avg=0.3902842483520145,	AK=0.8902842483520145

GS:Alpha=100.0,Iter=5000.0:Tr-ACC=0.8153153153153153,Tr-Kappa=0.7715949084384666.Va-Acc=0.5,Va-Kappa=0.390200138026225
GS:Alpha=100.0,Iter=5000.0:Tr-ACC=0.8175675675675675,Tr-Kappa=0.7740798673275163.Va-Acc=0.5,Va-Kappa=0.390200138026225
GS:Alpha=100.0,Iter=5000.0:Tr-ACC=0.8108108108108109,Tr-Kappa=0.7660240525467219.Va-Acc=0.5,Va-Kappa=0.390200138026225
GS:Alpha=100.0,Iter=5000.0:Tr-ACC=0.8198198198198198,Tr-Kappa=0.7769474708782065.Va-Acc=0.5,Va-Kappa=0.390200138026225
GS:Alpha=100.0,Iter=5000.0:Tr-ACC=0.8108108108108109,Tr-Kappa=0.7660035009128725.Va-Acc=0.5,Va-Kappa=0.390200138026225
GS:Alpha=100.0,Iter=5000.0:Tr-ACC=0.8130630630630631,Tr-Kappa=0.7689051653320124.Va-Acc=0.5,Va-Kappa=0.390200138026225
GS:Alpha=100.0,Iter=5000.0:Tr-ACC=0.8130630630630631,Tr-Kappa=0.7688848751669771.Va-Acc=0.5,Va-Kappa=0.390200138026225
GS:Alpha=100.0,Iter=5000.0:Tr-ACC=0.8175675675675675,Tr-Kappa=0.7742670457755098.Va-Acc=0.5,Va-Kappa=0.390200138026225
GS:Alpha=100.0,Iter=5000.0:Tr-ACC=0.8175675675675675,Tr-Kappa=0.7742670457755098.Va-Acc=0.5,Va-Kappa=0.390200138026225
GS:Alpha=100.0,Iter=5000.0:Tr-ACC=0.8153153153153153,Tr-Kappa=0.7715949084384666.Va-Acc=0.5,Va-Kappa=0.390200138026225
GS:Alpha=100.0,Iter=5000.0:
	Tr-ACC-Avg=0.81509009009009,	Tr-Kappa-Avg=0.771256884059226,		AK=1.5863469741493161
	Va-Acc-Avg=0.5,					Va-Kappa-Avg=0.39020013802622505,	AK=0.890200138026225

GS:Alpha=10,Iter=5000.0:Tr-ACC=0.8198198198198198,Tr-Kappa=0.7782176239564926.Va-Acc=0.5,Va-Kappa=0.38834279385296966
GS:Alpha=10,Iter=5000.0:Tr-ACC=0.8153153153153153,Tr-Kappa=0.7724755964953942.Va-Acc=0.5,Va-Kappa=0.38876591034864416
GS:Alpha=10,Iter=5000.0:Tr-ACC=0.8153153153153153,Tr-Kappa=0.772709962979842.Va-Acc=0.5,Va-Kappa=0.3875797061269753
GS:Alpha=10,Iter=5000.0:Tr-ACC=0.8153153153153153,Tr-Kappa=0.7727312450842083.Va-Acc=0.5,Va-Kappa=0.38834279385296966
GS:Alpha=10,Iter=5000.0:Tr-ACC=0.8175675675675675,Tr-Kappa=0.7753373313343328.Va-Acc=0.5,Va-Kappa=0.38715494520737964
GS:Alpha=10,Iter=5000.0:Tr-ACC=0.8175675675675675,Tr-Kappa=0.7756974372727442.Va-Acc=0.5,Va-Kappa=0.38876591034864416
GS:Alpha=10,Iter=5000.0:Tr-ACC=0.8153153153153153,Tr-Kappa=0.772709962979842.Va-Acc=0.5,Va-Kappa=0.38876591034864416
GS:Alpha=10,Iter=5000.0:Tr-ACC=0.8198198198198198,Tr-Kappa=0.7782176239564926.Va-Acc=0.5,Va-Kappa=0.38834279385296966
GS:Alpha=10,Iter=5000.0:Tr-ACC=0.8153153153153153,Tr-Kappa=0.7724755964953942.Va-Acc=0.5106382978723404,Va-Kappa=0.3999444907021925
GS:Alpha=10,Iter=5000.0:Tr-ACC=0.8130630630630631,Tr-Kappa=0.7698490516546862.Va-Acc=0.5,Va-Kappa=0.38901949937767943
GS:Alpha=10,Iter=5000.0:
	Tr-ACC-Avg=0.8164414414414415,	Tr-Kappa-Avg=0.7740421432209429,	AK=1.5904835846623844
	Va-Acc-Avg=0.5010638297872341,	Va-Kappa-Avg=0.38950247540190686,	AK=0.8905663051891409

GS:Alpha=100.0,Iter=7000.0:Tr-ACC=0.8333333333333334,Tr-Kappa=0.7936232757970905.Va-Acc=0.4787234042553192,Va-Kappa=0.36843548608254495
GS:Alpha=100.0,Iter=7000.0:Tr-ACC=0.8265765765765766,Tr-Kappa=0.7853954942344029.Va-Acc=0.46808510638297873,Va-Kappa=0.3559879419018909
GS:Alpha=100.0,Iter=7000.0:Tr-ACC=0.831081081081081,Tr-Kappa=0.7909210774157092.Va-Acc=0.4787234042553192,Va-Kappa=0.36843548608254495
GS:Alpha=100.0,Iter=7000.0:Tr-ACC=0.8355855855855856,Tr-Kappa=0.7963801757769555.Va-Acc=0.4787234042553192,Va-Kappa=0.36843548608254495
GS:Alpha=100.0,Iter=7000.0:Tr-ACC=0.831081081081081,Tr-Kappa=0.7909696372413014.Va-Acc=0.46808510638297873,Va-Kappa=0.3559879419018909
GS:Alpha=100.0,Iter=7000.0:Tr-ACC=0.831081081081081,Tr-Kappa=0.7909210774157092.Va-Acc=0.4787234042553192,Va-Kappa=0.36843548608254495
GS:Alpha=100.0,Iter=7000.0:Tr-ACC=0.8333333333333334,Tr-Kappa=0.7936232757970905.Va-Acc=0.4787234042553192,Va-Kappa=0.36843548608254495
GS:Alpha=100.0,Iter=7000.0:Tr-ACC=0.831081081081081,Tr-Kappa=0.7909250155393569.Va-Acc=0.4787234042553192,Va-Kappa=0.36695986805937325
GS:Alpha=100.0,Iter=7000.0:Tr-ACC=0.8333333333333334,Tr-Kappa=0.7936090154717859.Va-Acc=0.46808510638297873,Va-Kappa=0.3559879419018909
GS:Alpha=100.0,Iter=7000.0:Tr-ACC=0.831081081081081,Tr-Kappa=0.7909394540569047.Va-Acc=0.46808510638297873,Va-Kappa=0.3559879419018909
GS:Alpha=100.0,Iter=7000.0:
	Tr-ACC-Avg=0.8317567567567566,	Tr-Kappa-Avg=0.7917307498746307,	AK=1.6234875066313874
	Va-Acc-Avg=0.474468085106383,	Va-Kappa-Avg=0.36330890660796616,	AK=0.8377769917143492

GS:Alpha=1000.0,Iter=9000.0:Tr-ACC=0.8423423423423423,Tr-Kappa=0.8050481734243277.Va-Acc=0.43617021276595747,Va-Kappa=0.3162228932198738
GS:Alpha=1000.0,Iter=9000.0:Tr-ACC=0.8288288288288288,Tr-Kappa=0.7887646639039475.Va-Acc=0.43617021276595747,Va-Kappa=0.3178145967410653
GS:Alpha=1000.0,Iter=9000.0:Tr-ACC=0.8400900900900901,Tr-Kappa=0.8023945339434589.Va-Acc=0.44680851063829785,Va-Kappa=0.33196665299986333
GS:Alpha=1000.0,Iter=9000.0:Tr-ACC=0.8378378378378378,Tr-Kappa=0.799721835883171.Va-Acc=0.44680851063829785,Va-Kappa=0.33196665299986333
GS:Alpha=1000.0,Iter=9000.0:Tr-ACC=0.8400900900900901,Tr-Kappa=0.8023635622707752.Va-Acc=0.44680851063829785,Va-Kappa=0.33196665299986333
GS:Alpha=1000.0,Iter=9000.0:Tr-ACC=0.8423423423423423,Tr-Kappa=0.805061623859253.Va-Acc=0.44680851063829785,Va-Kappa=0.33196665299986333
GS:Alpha=1000.0,Iter=9000.0:Tr-ACC=0.8400900900900901,Tr-Kappa=0.8023586059021574.Va-Acc=0.43617021276595747,Va-Kappa=0.3178145967410653
GS:Alpha=1000.0,Iter=9000.0:Tr-ACC=0.8355855855855856,Tr-Kappa=0.7970241226422184.Va-Acc=0.43617021276595747,Va-Kappa=0.3178145967410653
GS:Alpha=1000.0,Iter=9000.0:Tr-ACC=0.8355855855855856,Tr-Kappa=0.7968103513127209.Va-Acc=0.43617021276595747,Va-Kappa=0.3162228932198738
GS:Alpha=1000.0,Iter=9000.0:Tr-ACC=0.831081081081081,Tr-Kappa=0.7914291798719763.Va-Acc=0.43617021276595747,Va-Kappa=0.3178145967410653
GS:Alpha=1000.0,Iter=9000.0:
	Tr-ACC-Avg=0.8373873873873874,	Tr-Kappa-Avg=0.7990976653014007,	AK=1.6364850526887877
	Va-Acc-Avg=0.4404255319148936,	Va-Kappa-Avg=0.3231570785403462,	AK=0.7635826104552398
"""

"""
GS-LRC-OvR
	GS:Alpha=1000.0,Iter=5000.0:Tr-ACC=0.6666666666666666,Tr-Kappa=0.5898357770162724.Va-Acc=0.2978723404255319,Va-Kappa=0.153037542662116
	GS:Alpha=1000.0,Iter=5000.0:Tr-ACC=0.6576576576576577,Tr-Kappa=0.581290598767845.Va-Acc=0.3191489361702128,Va-Kappa=0.17971093536951185
	GS:Alpha=1000.0,Iter=5000.0:Tr-ACC=0.6621621621621622,Tr-Kappa=0.5882763864761775.Va-Acc=0.32978723404255317,Va-Kappa=0.18204419889502757
	GS:Alpha=1000.0,Iter=5000.0:Tr-ACC=0.6531531531531531,Tr-Kappa=0.5753201744034384.Va-Acc=0.3404255319148936,Va-Kappa=0.19691332506545406
	GS:Alpha=1000.0,Iter=5000.0:Tr-ACC=0.6373873873873874,Tr-Kappa=0.5578920025481016.Va-Acc=0.32978723404255317,Va-Kappa=0.18000553863195787
	GS:Alpha=1000.0,Iter=5000.0:Tr-ACC=0.6576576576576577,Tr-Kappa=0.5829156598211472.Va-Acc=0.32978723404255317,Va-Kappa=0.18474669603524227
	GS:Alpha=1000.0,Iter=5000.0:Tr-ACC=0.6509009009009009,Tr-Kappa=0.5738251085253556.Va-Acc=0.2553191489361702,Va-Kappa=0.10219675262655202
	GS:Alpha=1000.0,Iter=5000.0:Tr-ACC=0.6418918918918919,Tr-Kappa=0.5639395904753081.Va-Acc=0.2872340425531915,Va-Kappa=0.14102564102564108
	GS:Alpha=1000.0,Iter=5000.0:Tr-ACC=0.6554054054054054,Tr-Kappa=0.580829677347698.Va-Acc=0.2765957446808511,Va-Kappa=0.1321113374066531
	GS:Alpha=1000.0,Iter=5000.0:Tr-ACC=0.6418918918918919,Tr-Kappa=0.5632165417906662.Va-Acc=0.3404255319148936,Va-Kappa=0.20664307105907972
	GS:Alpha=1000.0,Iter=5000.0:Tr-ACC-Avg=0.6524774774774775,Tr-Kappa-Avg=0.575734151717201.Va-Acc-Avg=0.31063829787234043,Va-Kappa-Avg=0.16584350387772354
	Tr-AK-Avg=1.228,Va-AK-Avg=0.47648180175006394
	
	GS:Alpha=100.0,Iter=5000.0:Tr-ACC=0.6283783783783784,Tr-Kappa=0.5502320669924609.Va-Acc=0.2765957446808511,Va-Kappa=0.12951109900585597
	GS:Alpha=100.0,Iter=5000.0:Tr-ACC=0.6441441441441441,Tr-Kappa=0.5676728334956183.Va-Acc=0.32978723404255317,Va-Kappa=0.1983213753891972
	GS:Alpha=100.0,Iter=5000.0:Tr-ACC=0.6509009009009009,Tr-Kappa=0.5748965346840447.Va-Acc=0.2978723404255319,Va-Kappa=0.15763747454175153
	GS:Alpha=100.0,Iter=5000.0:Tr-ACC=0.6441441441441441,Tr-Kappa=0.5674969173859433.Va-Acc=0.2978723404255319,Va-Kappa=0.16082781009062627
	GS:Alpha=100.0,Iter=5000.0:Tr-ACC=0.6441441441441441,Tr-Kappa=0.5658803435728607.Va-Acc=0.32978723404255317,Va-Kappa=0.18698517298187803
	GS:Alpha=100.0,Iter=5000.0:Tr-ACC=0.6261261261261262,Tr-Kappa=0.5455728123015458.Va-Acc=0.3404255319148936,Va-Kappa=0.2041513041103373
	GS:Alpha=100.0,Iter=5000.0:Tr-ACC=0.6531531531531531,Tr-Kappa=0.577119320184797.Va-Acc=0.2978723404255319,Va-Kappa=0.15775183274504478
	GS:Alpha=100.0,Iter=5000.0:Tr-ACC=0.6463963963963963,Tr-Kappa=0.5708375403255596.Va-Acc=0.26595744680851063,Va-Kappa=0.10894353620002747
	GS:Alpha=100.0,Iter=5000.0:Tr-ACC=0.668918918918919,Tr-Kappa=0.5986101288398266.Va-Acc=0.3191489361702128,Va-Kappa=0.17735539450294002
	GS:Alpha=100.0,Iter=5000.0:Tr-ACC=0.6621621621621622,Tr-Kappa=0.5891399700183221.Va-Acc=0.2872340425531915,Va-Kappa=0.14090847087709726
	GS:Alpha=100.0,Iter=5000.0:Tr-ACC-Avg=0.6468468468468468,Tr-Kappa-Avg=0.5707458467800979.Va-Acc-Avg=0.30425531914893617,Va-Kappa-Avg=0.16223934704447557
	Tr-AK-Avg=1.218,Va-AK-Avg=0.466
	
	GS:Alpha=10,Iter=5000.0:Tr-ACC=0.6283783783783784,Tr-Kappa=0.5514574876476316.Va-Acc=0.26595744680851063,Va-Kappa=0.12457821568362801
	GS:Alpha=10,Iter=5000.0:Tr-ACC=0.6103603603603603,Tr-Kappa=0.5296725366773617.Va-Acc=0.2765957446808511,Va-Kappa=0.14304866604102429
	GS:Alpha=10,Iter=5000.0:Tr-ACC=0.5765765765765766,Tr-Kappa=0.49231835930372586.Va-Acc=0.32978723404255317,Va-Kappa=0.19853836784409254
	GS:Alpha=10,Iter=5000.0:Tr-ACC=0.6171171171171171,Tr-Kappa=0.5392419590152426.Va-Acc=0.2872340425531915,Va-Kappa=0.15959434214037896
	GS:Alpha=10,Iter=5000.0:Tr-ACC=0.6373873873873874,Tr-Kappa=0.5648728109420939.Va-Acc=0.3191489361702128,Va-Kappa=0.18416056414429077
	GS:Alpha=10,Iter=5000.0:Tr-ACC=0.6103603603603603,Tr-Kappa=0.5325263369301272.Va-Acc=0.2978723404255319,Va-Kappa=0.16275303643724695
	GS:Alpha=10,Iter=5000.0:Tr-ACC=0.6013513513513513,Tr-Kappa=0.5211610874837012.Va-Acc=0.3191489361702128,Va-Kappa=0.1909628832705756
	GS:Alpha=10,Iter=5000.0:Tr-ACC=0.6013513513513513,Tr-Kappa=0.5204218003405159.Va-Acc=0.2765957446808511,Va-Kappa=0.1350473612990528
	GS:Alpha=10,Iter=5000.0:Tr-ACC=0.5990990990990991,Tr-Kappa=0.5192320560628273.Va-Acc=0.26595744680851063,Va-Kappa=0.11815091774303194
	GS:Alpha=10,Iter=5000.0:Tr-ACC=0.6216216216216216,Tr-Kappa=0.5461807562437258.Va-Acc=0.3191489361702128,Va-Kappa=0.19367377027208152
	GS:Alpha=10,Iter=5000.0:Tr-ACC-Avg=0.6103603603603603,Tr-Kappa-Avg=0.5317085190646953.Va-Acc-Avg=0.2957446808510638,Va-Kappa-Avg=0.16105081248754033
	Tr-AK-Avg=1.142,Va-AK-Avg=0.457
	
	GS:Alpha=100.0,Iter=7000.0:Tr-ACC=0.6756756756756757,Tr-Kappa=0.6065210568099995.Va-Acc=0.2978723404255319,Va-Kappa=0.1540769020998091
	GS:Alpha=100.0,Iter=7000.0:Tr-ACC=0.6554054054054054,Tr-Kappa=0.5825221238938053.Va-Acc=0.30851063829787234,Va-Kappa=0.16552854411363016
	GS:Alpha=100.0,Iter=7000.0:Tr-ACC=0.668918918918919,Tr-Kappa=0.5969269913417241.Va-Acc=0.32978723404255317,Va-Kappa=0.19570827108515548
	GS:Alpha=100.0,Iter=7000.0:Tr-ACC=0.668918918918919,Tr-Kappa=0.5974912582560915.Va-Acc=0.3617021276595745,Va-Kappa=0.22516829234785
	GS:Alpha=100.0,Iter=7000.0:Tr-ACC=0.6711711711711712,Tr-Kappa=0.6006087229755956.Va-Acc=0.32978723404255317,Va-Kappa=0.18798848210612915
	GS:Alpha=100.0,Iter=7000.0:Tr-ACC=0.6644144144144144,Tr-Kappa=0.5922563467263281.Va-Acc=0.2765957446808511,Va-Kappa=0.11785812862268835
	GS:Alpha=100.0,Iter=7000.0:Tr-ACC=0.6824324324324325,Tr-Kappa=0.6123087212578726.Va-Acc=0.2872340425531915,Va-Kappa=0.13678728070175442
	GS:Alpha=100.0,Iter=7000.0:Tr-ACC=0.6554054054054054,Tr-Kappa=0.5819620681591611.Va-Acc=0.2872340425531915,Va-Kappa=0.14557047890381225
	GS:Alpha=100.0,Iter=7000.0:Tr-ACC=0.6756756756756757,Tr-Kappa=0.6060871172447785.Va-Acc=0.26595744680851063,Va-Kappa=0.09753721998052038
	GS:Alpha=100.0,Iter=7000.0:Tr-ACC=0.6644144144144144,Tr-Kappa=0.5923442853269576.Va-Acc=0.30851063829787234,Va-Kappa=0.15526061108806857
	GS:Alpha=100.0,Iter=7000.0:Tr-ACC-Avg=0.6682432432432432,Tr-Kappa-Avg=0.5969028691992313.Va-Acc-Avg=0.30531914893617024,Va-Kappa-Avg=0.1581484211049418
	Tr-AK-Avg=1.265,Va-AK-Avg=0.463
	
	GS:Alpha=1000.0,Iter=9000.0:Tr-ACC=0.6666666666666666,Tr-Kappa=0.5927539555147902.Va-Acc=0.3829787234042553,Va-Kappa=0.24403771491957846
	GS:Alpha=1000.0,Iter=9000.0:Tr-ACC=0.6621621621621622,Tr-Kappa=0.5879171874419928.Va-Acc=0.2978723404255319,Va-Kappa=0.14356709000552179
	GS:Alpha=1000.0,Iter=9000.0:Tr-ACC=0.6531531531531531,Tr-Kappa=0.5778190776678048.Va-Acc=0.3404255319148936,Va-Kappa=0.1960270382121672
	GS:Alpha=1000.0,Iter=9000.0:Tr-ACC=0.6599099099099099,Tr-Kappa=0.5852469563495992.Va-Acc=0.32978723404255317,Va-Kappa=0.18921139101861995
	GS:Alpha=1000.0,Iter=9000.0:Tr-ACC=0.6779279279279279,Tr-Kappa=0.6079312836156378.Va-Acc=0.3617021276595745,Va-Kappa=0.21426581220395657
	GS:Alpha=1000.0,Iter=9000.0:Tr-ACC=0.6621621621621622,Tr-Kappa=0.5886425822869249.Va-Acc=0.32978723404255317,Va-Kappa=0.18429752066115698
	GS:Alpha=1000.0,Iter=9000.0:Tr-ACC=0.6869369369369369,Tr-Kappa=0.6189194195739426.Va-Acc=0.32978723404255317,Va-Kappa=0.18227009113504553
	GS:Alpha=1000.0,Iter=9000.0:Tr-ACC=0.668918918918919,Tr-Kappa=0.5960438689872007.Va-Acc=0.30851063829787234,Va-Kappa=0.15221312612737617
	GS:Alpha=1000.0,Iter=9000.0:Tr-ACC=0.6734234234234234,Tr-Kappa=0.6023471278567016.Va-Acc=0.3829787234042553,Va-Kappa=0.2483110437060527
	GS:Alpha=1000.0,Iter=9000.0:Tr-ACC=0.6869369369369369,Tr-Kappa=0.6184081170818572.Va-Acc=0.2872340425531915,Va-Kappa=0.1240611961057024
	GS:Alpha=1000.0,Iter=9000.0:Tr-ACC-Avg=0.6698198198198199,Tr-Kappa-Avg=0.5976029576376453.Va-Acc-Avg=0.3351063829787234,Va-Kappa-Avg=0.18782620240951778
	Tr-AK-Avg=1.267,Va-AK-Avg=0.523
"""

def lrcFinalRes():
	"""
	此函数的目的是 计算 LRC-Final-OvO: (Linear α, Initial α = 100, Iteration = 5000) 在 Final的Acc 和 Kappa 以和AB的Final结果比较
	LRC
    	GS的最佳配置（OvO，α=1，iter=9K），LRC的GS分两次完成，第一次做了一个小网格（训练日期在2020-04-06 ~ 2020-04-11，训练的模型文件上有日期标注），
    	但是看到Heatmap的右侧结果（更大的alpha和更大的iter）较好，所以又加了一批（训练日期在2020-04-20 ~ 2020-04-22，训练的模型文件上有日期标注）

    Final的配置OvR-OvO，
        OvR: (Linear α, Initial α = 1, Iteration = 3000)
        OvO: (Linear α, Initial α = 1000, Iteration = 7000)

	AB
		AB-GS的配置（LRC，OvO，α=1，iter=9K），此时只知道LRC最佳的配置是（LRC：OVO，α=100，iter=5K），
		所以就用这个配置。后面LRC——GS又加了一批新的参数的尝试，发现（LRC-OVO，α = 1，iter=9K）更好
		Final的配置（LRC: (OvO, Linear α, Initial α = 1, Iteration = 9000)，Number = 150）
	"""
	modelAbsFp = './ovo_res/linear_final/models/2020_04_22_lrc_ovo_linear_final_iter=5000_alpha_init=100_classifer_dict_pickle_3.file'
	lrcFinal = LRC(alpha=0, max_iter=0,
				   unlabeled_dataset_list=tr_va_data_list,
				   labeled_dataset_list=tr_va_dataset,
				   label_list=label_list)
	
	# get res on trVa
	lrcFinalTrVaSample_label_prob_dict_list = lrcFinal.classify_ovo(lrc_classifer_dict_pickle_filename=modelAbsFp)
	lrcFinalTrVaAcc = cal_accuracy(lrcFinalTrVaSample_label_prob_dict_list, tr_va_label_list)
	lrcFinalTrVaKappa = cal_kappa(lrcFinalTrVaSample_label_prob_dict_list, tr_va_label_list)
	print('LRC-Final-TrVa: Acc={0}, Kappa={1}'.format(lrcFinalTrVaAcc, lrcFinalTrVaKappa))
	
	# get res on te
	lrcFinal.unlabeled_dataset_list = te_data_list
	lrcFinalTeSample_label_prob_dict_list = lrcFinal.classify_ovo(lrc_classifer_dict_pickle_filename=modelAbsFp)
	lrcFinalTeAcc = cal_accuracy(lrcFinalTeSample_label_prob_dict_list, te_label_list)
	lrcFinalTeKappa = cal_kappa(lrcFinalTeSample_label_prob_dict_list, te_label_list)
	print('LRC-Final-Te: Acc={0}, Kappa={1}'.format(lrcFinalTeAcc, lrcFinalTeKappa))
# lrcFinalRes()
"""
LRC-Final-TrVa: Acc=0.7769516728624535, Kappa=0.7244511216580735
LRC-Final-Te: Acc=0.4725274725274725, Kappa=0.3421686746987952
"""