import os

from ml_sl.ml_data_wrapper import split_labeled_dataset_list
from ml_sl.ml_critrions import cal_accuracy, cal_kappa, cal_accuracy_on_2, cal_accuracy_on_3
from ml_sl.svm.multiclass_svm_0 import Multiclass_SVM, load_Multiclass_SVM_model
from utils.file_utils.dataset_reader_pack.ml_dataset_reader import get_TV_T_dataset, get_T_V_T_dataset

"""
本模块用于产生EA-Revise时要求的结果
	1- 网格搜索时，SVM结合最佳的五组超参数，每组参数训练出10个模型,分别计算这10个模型的【训练=tr】 和 【测试=va】上的 Accuracy 和 Kappa
		SVM网格搜索的结果由于太多，放在dpfc项目以外，具体：
			NH55：D:\cs_0\git_prjs\dpfc_large_files\ml_sl\svm
	2- SVM结合最佳的超参数，分别计算【训练=tr+va】 和 【测试=te】上的 Accuracy 和 Kappa
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

def svmGridSearchRes():
	# top5HyperParaList=[[constant (C) = 10^3, Iteration time = 5k], [],[]]
	top5HyperParaList = [[1e-3, 3e3], [1e-3, 5e3], [1e-3, 9e3], [1e-3, 1e3], [1e-3, 7e3]]
	
	# 因为每组参数都被重复训练了10次，要从已经训练好的模型文件中，将每组参数对应的10个模型找出来
	# 所有GS-SVM-OvO的模型文件都存放在：D:\cs_0\git_prjs\dpfc_large_files\ml_sl\svm\ovo_models\trained_on_tr_tested_on_vali
	modelFp = '../../../../dpfc_large_files/ml_sl/svm/ovo_models/trained_on_tr_tested_on_vali/'
	fnList = os.listdir(modelFp)
	
	for hyperPara in top5HyperParaList:
		c, iterTime = hyperPara
		
		targetFnList = []
		for fn in fnList:
			fnStrList = fn.split('_')
			
			"""
			根据文件名中的【Linear/poly/rbf】选出 linear kernel 的 model file
				Linear
					2020_05_06_svm_linear_C=100_iter=5000_pickle_9.file
				poly
					2020_05_06_svm_poly_C=100_iter=3000_P=4_q=0.001_pickle_8.file
				rbf
					2020_05_04_svm_rbf_C=10000_iter=1000_sigma=1e-05_pickle_5.file
			"""
			kernelStr = fnStrList[4]
			if kernelStr == 'linear':
				c1 = float(fnStrList[5].split('=')[1])
				iterTime1 = float(fnStrList[6].split('=')[1])
				if (c == c1) and (iterTime == iterTime1):
					targetFnList.append(fn)
					
		# 逐个加载 model 并 进行 实验
		multiSvmGSTrAccList = []
		multiSvmGSTrKappaList = []
		multiSvmGSVaAccList = []
		multiSvmGSVaKappaList = []
		for targetFn in targetFnList:
			modelAbsFp = os.path.join(modelFp, targetFn)
			multiSvmGSTr = Multiclass_SVM(svm_para_dict={'C': c, 'max_iter': iterTime},
									   kernel_para_dict={'type': 'linear', 'paras': None},
									   unlabeled_dataset_list=tr_data_list,
									   labeled_dataset_list=tr_dataset,
									   label_list=label_list)
			
			# SVM classify Tr-dataset
			multiSvmGSTrSample_label_prob_dict_list = multiSvmGSTr.classify_ovo(svm_ovo_model_pickle_name=modelAbsFp)
			
			# get accuracy and kappa
			multiSvmGSTrAcc = cal_accuracy(multiSvmGSTrSample_label_prob_dict_list, tr_label_list)
			multiSvmGSTrAccList.append(multiSvmGSTrAcc)
			multiSvmGSTrKappa = cal_kappa(multiSvmGSTrSample_label_prob_dict_list, tr_label_list)
			multiSvmGSTrKappaList.append(multiSvmGSTrKappa)
			
			multiSvmGSVa = Multiclass_SVM(svm_para_dict={'C': c, 'max_iter': iterTime},
										  kernel_para_dict={'type': 'linear', 'paras': None},
										  unlabeled_dataset_list=va_data_list,
										  labeled_dataset_list=tr_dataset,
										  label_list=label_list)
			
			# SVM classify Tr-dataset
			multiSvmGSVaSample_label_prob_dict_list = multiSvmGSVa.classify_ovo(svm_ovo_model_pickle_name=modelAbsFp)
			
			# get accuracy and kappa
			multiSvmGSVaAcc = cal_accuracy(multiSvmGSVaSample_label_prob_dict_list, va_label_list)
			multiSvmGSVaAccList.append(multiSvmGSVaAcc)
			multiSvmGSVaKappa = cal_kappa(multiSvmGSVaSample_label_prob_dict_list, va_label_list)
			multiSvmGSVaKappaList.append(multiSvmGSVaKappa)
			
			print('GS:C={0},Iter={1}==>Tr-Acc={2},Tr-Kappa={3}==>Va-Acc={4},Va-Kappa={5}, AK-Tr={6}'.format(
				c, iterTime,
				multiSvmGSTrAcc, multiSvmGSTrKappa,
				multiSvmGSVaAcc, multiSvmGSVaKappa,
				multiSvmGSTrAcc + multiSvmGSTrKappa
			))
		print('GS:C={0},Iter={1}==>Tr-Acc-Avg={2},Tr-Kappa-Avg={3}==>Va-Acc-Avg={4},Va-Kappa-Avg={5}'.format(
			c, iterTime,
			sum(multiSvmGSTrAccList) / len(multiSvmGSTrAccList), sum(multiSvmGSTrKappaList) / len(multiSvmGSTrKappaList),
			sum(multiSvmGSVaAccList) / len(multiSvmGSVaAccList), sum(multiSvmGSVaKappaList) / len(multiSvmGSVaKappaList)
		))
# svmGridSearchRes()
"""
直接运行代码会提示找不到 模块 ml_sl，因为之前保存模型时，ml还叫ml_sl，所以将现在的整个项目拷贝出来，再把ml改成ml_sl，直接在拷贝的版本里可正常运行
D:\cs_0\python\install\python.exe D:/cs_0/git_prjs/distributed_parallel_fitting_circuit/dpfc_src1/ml_sl/svm/svmReviseRes_0.py
GS:C=0.001,Iter=3000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24421104862719153==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6879047423208853
GS:C=0.001,Iter=3000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24389840324315384==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6875920969368475
GS:C=0.001,Iter=3000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.2448355643695257==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6885292580632194
GS:C=0.001,Iter=3000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24452343556253622==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6882171292562299
GS:C=0.001,Iter=3000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24452343556253622==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6882171292562299
GS:C=0.001,Iter=3000.0==>Tr-Acc=0.44144144144144143,Tr-Kappa=0.24052309220327758==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.681964533644719
GS:C=0.001,Iter=3000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24389840324315384==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6875920969368475
GS:C=0.001,Iter=3000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24421104862719153==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6879047423208853
GS:C=0.001,Iter=3000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24514743536796899==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6888411290616627
GS:C=0.001,Iter=3000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24389840324315384==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6875920969368475
GS:C=0.001,Iter=3000.0	==> Tr-Acc-Avg=0.4434684684684685,Tr-Kappa-Avg=0.24396702700496892,AK=0.6874354954734374
					 	==> Va-Acc-Avg=0.44680851063829785,Va-Kappa-Avg=0.24799999999999994,AK=0.6948085106382977

GS:C=0.001,Iter=5000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24389840324315384==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6875920969368475
GS:C=0.001,Iter=5000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24389840324315384==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6875920969368475
GS:C=0.001,Iter=5000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.2448355643695257==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6885292580632194
GS:C=0.001,Iter=5000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24421104862719153==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6879047423208853
GS:C=0.001,Iter=5000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24389840324315384==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6875920969368475
GS:C=0.001,Iter=5000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24452343556253622==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6882171292562299
GS:C=0.001,Iter=5000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24452343556253622==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6882171292562299
GS:C=0.001,Iter=5000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24526172353516032==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.688955417228854
GS:C=0.001,Iter=5000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24421104862719153==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6879047423208853
GS:C=0.001,Iter=5000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.2448355643695257==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6885292580632194
GS:C=0.001,Iter=5000.0	==>	Tr-Acc-Avg=0.4436936936936937,Tr-Kappa-Avg=0.24440970303831291,AK=0.6881033967320066
						==>	Va-Acc-Avg=0.44680851063829785,Va-Kappa-Avg=0.24799999999999994,AK=0.6948085106382977

GS:C=0.001,Iter=9000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.2448355643695257==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6885292580632194
GS:C=0.001,Iter=9000.0==>Tr-Acc=0.44144144144144143,Tr-Kappa=0.24083726317531232==>Va-Acc=0.44680851063829785,Va-Kappa=0.24649298597194386, AK-Tr=0.6822787046167538
GS:C=0.001,Iter=9000.0==>Tr-Acc=0.44144144144144143,Tr-Kappa=0.24052309220327758==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.681964533644719
GS:C=0.001,Iter=9000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.2448355643695257==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6885292580632194
GS:C=0.001,Iter=9000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.2448355643695257==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6885292580632194
GS:C=0.001,Iter=9000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24389840324315384==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6875920969368475
GS:C=0.001,Iter=9000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24452343556253622==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6882171292562299
GS:C=0.001,Iter=9000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24452343556253622==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6882171292562299
GS:C=0.001,Iter=9000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24514743536796899==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6888411290616627
GS:C=0.001,Iter=9000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24452343556253622==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6882171292562299
GS:C=0.001,Iter=9000.0	==>	Tr-Acc-Avg=0.44324324324324327,Tr-Kappa-Avg=0.24384831937858986,AK=0.6870915626218331
						==>	Va-Acc-Avg=0.44680851063829785,Va-Kappa-Avg=0.24784929859719435,AK=0.6946578092354923

GS:C=0.001,Iter=1000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24452343556253622==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6882171292562299
GS:C=0.001,Iter=1000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.2448355643695257==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6885292580632194
GS:C=0.001,Iter=1000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24452343556253622==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6882171292562299
GS:C=0.001,Iter=1000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24389840324315384==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6875920969368475
GS:C=0.001,Iter=1000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24514743536796899==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6888411290616627
GS:C=0.001,Iter=1000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24452343556253622==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6882171292562299
GS:C=0.001,Iter=1000.0==>Tr-Acc=0.44594594594594594,Tr-Kappa=0.2495980873078033==>Va-Acc=0.43617021276595747,Va-Kappa=0.2368259803921569, AK-Tr=0.6955440332537492
GS:C=0.001,Iter=1000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24389840324315384==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6875920969368475
GS:C=0.001,Iter=1000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24432562049529377==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6880193141889874
GS:C=0.001,Iter=1000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24452343556253622==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6882171292562299
GS:C=0.001,Iter=1000.0	==>	Tr-Acc-Avg=0.44391891891891894,Tr-Kappa-Avg=0.24497972562770448,AK=0.6888986445466234
						==>	Va-Acc-Avg=0.4457446808510638,Va-Kappa-Avg=0.2468825980392156,AK=0.6926272788902794

GS:C=0.001,Iter=7000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24421104862719153==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6879047423208853
GS:C=0.001,Iter=7000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24514743536796899==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6888411290616627
GS:C=0.001,Iter=7000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.2423312883435583==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.686024982037252
GS:C=0.001,Iter=7000.0==>Tr-Acc=0.44144144144144143,Tr-Kappa=0.24146482598991487==>Va-Acc=0.44680851063829785,Va-Kappa=0.24649298597194386, AK-Tr=0.6829062674313563
GS:C=0.001,Iter=7000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24452343556253622==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6882171292562299
GS:C=0.001,Iter=7000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24421104862719153==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6879047423208853
GS:C=0.001,Iter=7000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24452343556253622==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6882171292562299
GS:C=0.001,Iter=7000.0==>Tr-Acc=0.44144144144144143,Tr-Kappa=0.2411511743301356==>Va-Acc=0.44680851063829785,Va-Kappa=0.24649298597194386, AK-Tr=0.682592615771577
GS:C=0.001,Iter=7000.0==>Tr-Acc=0.44144144144144143,Tr-Kappa=0.24052309220327758==>Va-Acc=0.43617021276595747,Va-Kappa=0.22891193313728528, AK-Tr=0.681964533644719
GS:C=0.001,Iter=7000.0==>Tr-Acc=0.4436936936936937,Tr-Kappa=0.24389840324315384==>Va-Acc=0.44680851063829785,Va-Kappa=0.24799999999999994, AK-Tr=0.6875920969368475
GS:C=0.001,Iter=7000.0	==>	Tr-Acc-Avg=0.443018018018018,Tr-Kappa-Avg=0.24319851878574647,AK=0.6862165368037645
						==>	Va-Acc-Avg=0.4457446808510638,Va-Kappa-Avg=0.24578979050811722,AK=0.691534471359181
"""

def svmFinalRes():
	# 2020_06_26_svm_ovo_linear_final_C=0.001_iter=5000_3_pickle.file
	modelFp = './ovo_models/final/2020_06_26_svm_ovo_linear_final_C=0.001_iter=5000_3_pickle.file'
	
	multiSvmGSTrVa = Multiclass_SVM(svm_para_dict={'C': 0.001, 'max_iter': 5000},
									kernel_para_dict={'type': 'linear', 'paras': None},
									unlabeled_dataset_list=tr_va_data_list,
									labeled_dataset_list=tr_va_dataset,
									label_list=label_list)
	
	# SVM classify TrVa-dataset
	multiSvmGSTrVaSample_label_prob_dict_list = multiSvmGSTrVa.classify_ovo(svm_ovo_model_pickle_name=modelFp)
	
	# get accuracy and kappa
	multiSvmGSTrVaAcc = cal_accuracy(multiSvmGSTrVaSample_label_prob_dict_list, tr_va_label_list)
	multiSvmGSTrVaKappa = cal_kappa(multiSvmGSTrVaSample_label_prob_dict_list, tr_va_label_list)
	print('Final model Res on TrVa:Acc={0}, Kappa={1}'.format(multiSvmGSTrVaAcc, multiSvmGSTrVaKappa))
	
	# ------------------------------------------
	# SVM classify Te-dataset
	multiSvmGSTrVa.unlabeled_dataset_list = te_data_list
	multiSvmGSTeSample_label_prob_dict_list = multiSvmGSTrVa.classify_ovo(svm_ovo_model_pickle_name=modelFp)
	
	# get accuracy and kappa
	multiSvmGSTeAcc = cal_accuracy(multiSvmGSTeSample_label_prob_dict_list, te_label_list)
	multiSvmGSTeKappa = cal_kappa(multiSvmGSTeSample_label_prob_dict_list, te_label_list)
	print('Final model Res on Te:Acc={0}, Kappa={1}'.format(multiSvmGSTeAcc, multiSvmGSTeKappa))
# svmFinalRes()
# Final model Res on TrVa:	Acc=0.44609665427509293, Kappa=0.24612657346932057, AK=0.6922232277444135
# Final model Res on Te:	Acc=0.46153846153846156, Kappa=0.24755315558555518, AK=0.7090916171240167