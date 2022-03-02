import copy
import os

from utils.file_utils.dataset_reader_pack.ml_dataset_reader import get_TV_T_dataset, get_T_V_T_dataset
from ml_sl.rf.dt_0 import Node, save_node, load_node
from ml_sl.rf.rf_0 import RF, save_random_forest, load_random_forest
from ml_sl.ml_data_wrapper import pack_list_2_list, single_point_list_2_list, reform_labeled_dataset_list
from ml_sl.ml_data_wrapper import split_labeled_dataset_list
from ml_sl.ml_critrions import cal_accuracy, cal_kappa, cal_accuracy_on_2, cal_accuracy_on_3

"""
本模块用于产生EA-Revise时要求的结果
	1- 网格搜索时，AB结合最佳的五组超参数，每组参数训练出10个模型,分别计算这10个模型的【训练=tr】 和 【测试=va】上的 Accuracy 和 Kappa
		RF网格搜索的结果由于太多，放在dpfc项目以外
	2- RF合最佳的超参数，分别计算【训练=tr+va】 和 【测试=te】上的 Accuracy 和 Kappa
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

def rfGridSearchRes():
	# top5HyperParaList = [Base Learner Number, ...]
	top5HyperParaList = [200, 400, 300, 450, 350]
	
	# NH55 D:\cs_0\git_prjs\dpfc_large_files\ml_sl\rf\trained_on_tr_tested_on_vali
	modelFp = '../../../../dpfc_large_files/ml_sl/rf/trained_on_tr_tested_on_vali/'
	fnList = os.listdir(modelFp)
	
	for hyper in top5HyperParaList:
		targetFnList = []
		for fn in fnList:
			# 2020_04_14_rf_tree=50_pickle_2.file
			treeNum = float(fn.split('_')[4].split('=')[1])
			if hyper == treeNum:
				targetFnList.append(fn)
		
		rfGsTrAccList = []
		rfGsTrKappaList = []
		rfGsVaAccList = []
		rfGsVaKappaList = []
		for targetFn in targetFnList:
			rfGS = load_random_forest(filename=targetFn,
									  filepath=modelFp)
			
			# res on tr-data
			rfGS.unlabeled_dataset_list = tr_data_list
			rfGsTrSample_label_prob_dict_list= rfGS.classify()
			rfGsTrAcc = cal_accuracy(rfGsTrSample_label_prob_dict_list, tr_label_list)
			rfGsTrAccList.append(rfGsTrAcc)
			rfGsTrKappa = cal_kappa(rfGsTrSample_label_prob_dict_list, tr_label_list)
			rfGsTrKappaList.append(rfGsTrKappa)
			
			# res on va-data
			rfGS.unlabeled_dataset_list = va_data_list
			rfGsVaSample_label_prob_dict_list= rfGS.classify()
			rfGsVaAcc = cal_accuracy(rfGsVaSample_label_prob_dict_list, va_label_list)
			rfGsVaAccList.append(rfGsVaAcc)
			rfGsVaKappa = cal_kappa(rfGsVaSample_label_prob_dict_list, va_label_list)
			rfGsVaKappaList.append(rfGsVaKappa)
			
			print('RF GS: Base learner number={0} ==> Tr-Acc={1},Tr-Kappa={2}==>Va-Acc={3},Va-Kappa={4}'.format(
				hyper,
				rfGsTrAcc, rfGsTrKappa,
				rfGsVaAcc, rfGsVaKappa
				))
		print('RF GS: Base learner number={0} ==> Tr-Acc-Avg={1},Tr-Kappa-Avg={2} ==> Va-Acc-Avg={3},Va-Kappa-Avg={4}'.format(
			hyper,
			sum(rfGsTrAccList) / len(rfGsTrAccList),
			sum(rfGsTrKappaList) / len(rfGsTrKappaList),
			sum(rfGsVaAccList) / len(rfGsVaAccList),
			sum(rfGsVaKappaList) / len(rfGsVaKappaList)
		))
# rfGridSearchRes()
"""
ModuleNotFoundError: No module named 'ml_sl'
D:\cs_0\python\install\python.exe D:/cs_0/git_prjs/distributed_parallel_fitting_circuit/dpfc_src1/ml_sl/rf/rfReviseRes_0.py
RF GS: Base learner number=200 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5425531914893617,Va-Kappa=0.4256075031973852
RF GS: Base learner number=200 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.574468085106383,Va-Kappa=0.4643874643874645
RF GS: Base learner number=200 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5638297872340425,Va-Kappa=0.454339515786493
RF GS: Base learner number=200 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5531914893617021,Va-Kappa=0.4382470119521913
RF GS: Base learner number=200 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5531914893617021,Va-Kappa=0.4339784946236559
RF GS: Base learner number=200 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5425531914893617,Va-Kappa=0.42495376298193194
RF GS: Base learner number=200 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.574468085106383,Va-Kappa=0.466363894408175
RF GS: Base learner number=200 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5212765957446809,Va-Kappa=0.39692044482463645
RF GS: Base learner number=200 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5851063829787234,Va-Kappa=0.477703376549366
RF GS: Base learner number=200 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5638297872340425,Va-Kappa=0.4527900042595485
RF GS: Base learner number=200 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5531914893617021,Va-Kappa=0.4426090639559509
RF GS: Base learner number=200 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5531914893617021,Va-Kappa=0.4351123193589927
RF GS: Base learner number=200 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5638297872340425,Va-Kappa=0.45185606599345757
RF GS: Base learner number=200 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.574468085106383,Va-Kappa=0.46719569222049034
RF GS: Base learner number=200 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.574468085106383,Va-Kappa=0.4646162608571836
RF GS: Base learner number=200 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5851063829787234,Va-Kappa=0.47822374039282667
RF GS: Base learner number=200 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5531914893617021,Va-Kappa=0.43784707390004274
RF GS: Base learner number=200
		==> Tr-Acc-Avg=1.0,					Tr-Kappa-Avg=1.0,				AK=2
		==> Va-Acc-Avg=0.5607008760951189,	Va-Kappa-Avg=0.447808922920576,	AK=1.0085097990156948

RF GS: Base learner number=400 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5425531914893617,Va-Kappa=0.4214970659796764
RF GS: Base learner number=400 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5638297872340425,Va-Kappa=0.4499000856408792
RF GS: Base learner number=400 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5638297872340425,Va-Kappa=0.45076243408864186
RF GS: Base learner number=400 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5531914893617021,Va-Kappa=0.4368848951647411
RF GS: Base learner number=400 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5531914893617021,Va-Kappa=0.4392045454545455
RF GS: Base learner number=400 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5638297872340425,Va-Kappa=0.44950721325524917
RF GS: Base learner number=400 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5638297872340425,Va-Kappa=0.45154404440017076
RF GS: Base learner number=400 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5638297872340425,Va-Kappa=0.44721744119334483
RF GS: Base learner number=400 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.574468085106383,Va-Kappa=0.4624731951393853
RF GS: Base learner number=400 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.574468085106383,Va-Kappa=0.4646162608571836
RF GS: Base learner number=400
		==> Tr-Acc-Avg=1.0,					Tr-Kappa-Avg=1.0,					AK=2
		==> Va-Acc-Avg=0.5617021276595745,	Va-Kappa-Avg=0.44736071811738176,	AK=1.0090628457769562

RF GS: Base learner number=300 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5531914893617021,Va-Kappa=0.4375267132070096
RF GS: Base learner number=300 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5638297872340425,Va-Kappa=0.4491925110761755
RF GS: Base learner number=300 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5425531914893617,Va-Kappa=0.4230659434770196
RF GS: Base learner number=300 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5425531914893617,Va-Kappa=0.4238882554161915
RF GS: Base learner number=300 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5531914893617021,Va-Kappa=0.4384867017493956
RF GS: Base learner number=300 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5638297872340425,Va-Kappa=0.4525568181818182
RF GS: Base learner number=300 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.574468085106383,Va-Kappa=0.4641584722816019
RF GS: Base learner number=300 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5638297872340425,Va-Kappa=0.4541849596374451
RF GS: Base learner number=300 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5638297872340425,Va-Kappa=0.4501355400199743
RF GS: Base learner number=300 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5638297872340425,Va-Kappa=0.4505275163957798
RF GS: Base learner number=300
		==> Tr-Acc-Avg=1.0,					Tr-Kappa-Avg=1.0,					AK=2
		==> Va-Acc-Avg=0.5585106382978723,	Va-Kappa-Avg=0.4443723431442411,	AK=1.0028829814421134

RF GS: Base learner number=450 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5531914893617021,Va-Kappa=0.43551615670574784
RF GS: Base learner number=450 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5531914893617021,Va-Kappa=0.4343839541547278
RF GS: Base learner number=450 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5425531914893617,Va-Kappa=0.4230659434770196
RF GS: Base learner number=450 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5425531914893617,Va-Kappa=0.4219107551487413
RF GS: Base learner number=450 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5638297872340425,Va-Kappa=0.4499000856408792
RF GS: Base learner number=450 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5531914893617021,Va-Kappa=0.43736639589568194
RF GS: Base learner number=450 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5319148936170213,Va-Kappa=0.4107422709787719
RF GS: Base learner number=450 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5531914893617021,Va-Kappa=0.43768693918245266
RF GS: Base learner number=450 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5638297872340425,Va-Kappa=0.4505275163957798
RF GS: Base learner number=450 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5638297872340425,Va-Kappa=0.4472174411933448
RF GS: Base learner number=450
		==> Tr-Acc-Avg=1.0,					Tr-Kappa-Avg=1.0,					AK=2
		==> Va-Acc-Avg=0.5521276595744681,	Va-Kappa-Avg=0.4348317458773147,	AK=0.9869594054517827

RF GS: Base learner number=350 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5531914893617021,Va-Kappa=0.4381670698733457
RF GS: Base learner number=350 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5531914893617021,Va-Kappa=0.437125748502994
RF GS: Base learner number=350 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5638297872340425,Va-Kappa=0.44903502501786985
RF GS: Base learner number=350 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5425531914893617,Va-Kappa=0.42487194080819574
RF GS: Base learner number=350 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5531914893617021,Va-Kappa=0.43888573052870955
RF GS: Base learner number=350 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5531914893617021,Va-Kappa=0.43760683760683766
RF GS: Base learner number=350 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5425531914893617,Va-Kappa=0.42355961209355386
RF GS: Base learner number=350 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5638297872340425,Va-Kappa=0.4516220830961867
RF GS: Base learner number=350 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5212765957446809,Va-Kappa=0.39666238767650835
RF GS: Base learner number=350 ==> Tr-Acc=1.0,Tr-Kappa=1.0==>Va-Acc=0.5531914893617021,Va-Kappa=0.43704548695280193
RF GS: Base learner number=350
		==> Tr-Acc-Avg=1.0,		Tr-Kappa-Avg=1.0,					AK=2
		==> Va-Acc-Avg=0.55,	Va-Kappa-Avg=0.4334581922157004,	AK=0.9834581922157004
"""

def rfFinalRes():
	finalRF = load_random_forest(filename='2020_06_29_rf_tree=200_pickle_0.file',
								 # Desktop
								 # filepath='../../../../large_files/dpfc/ml_sl/rf/rf_res/trained_on_TV_tested_on_test')
								 # NH55 D:\cs_0\git_prjs\dpfc_large_files\ml_sl\rf\rf_res\trained_on_TV_tested_on_test
								 filepath='../../../../dpfc_large_files/ml_sl/rf/rf_res/trained_on_TV_tested_on_test')

	# res on trVa-data
	finalRF.unlabeled_dataset_list = tr_va_data_list
	rfTrVaSample_label_prob_dict_list = finalRF.classify()
	rfTrVaAcc = cal_accuracy(rfTrVaSample_label_prob_dict_list, tr_va_label_list)
	rfTrVaKappa = cal_kappa(rfTrVaSample_label_prob_dict_list, tr_va_label_list)
	
	# res on te-data
	finalRF.unlabeled_dataset_list = te_data_list
	rfTeSample_label_prob_dict_list = finalRF.classify()
	rfTeAcc = cal_accuracy(rfTeSample_label_prob_dict_list, te_label_list)
	rfTeKappa = cal_kappa(rfTeSample_label_prob_dict_list, te_label_list)
	
	print('Final RF: rfTrVaAcc={0}, rfTrVaKappa={1}, rfTeAcc={2}, rfTeKappa={3}'.format(
		rfTrVaAcc,rfTrVaKappa,
		rfTeAcc,rfTeKappa
	))
# rfFinalRes()
"""
Final RF:
	rfTrVaAcc=0.9962825278810409, 	rfTrVaKappa=0.9954208089336782, 	rfTrVaAK=1.991703336814719
	rfTeAcc=0.5824175824175825, 	rfTeKappa=0.46010928961748637, 		rfTeAK=1.0425268720350689
"""
