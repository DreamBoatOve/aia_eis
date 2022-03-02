import os

from ml_sl.adaboost.ab_0 import AB
from ml_sl.ml_data_wrapper import split_labeled_dataset_list
from ml_sl.ml_critrions import cal_accuracy, cal_kappa, cal_accuracy_on_2, cal_accuracy_on_3
from utils.file_utils.dataset_reader_pack.ml_dataset_reader import get_TV_T_dataset, get_T_V_T_dataset

"""
本模块用于产生EA-Revise时要求的结果
	1- 网格搜索时，AB结合最佳的五组超参数，每组参数训练出10个模型,分别计算这10个模型的【训练=tr】 和 【测试=va】上的 Accuracy 和 Kappa
		AB网格搜索的结果由于太多，放在dpfc项目以外，具体：
			NH55：D:\cs_0\git_prjs\dpfc_large_files\ml_sl\ab
	2- AB合最佳的超参数，分别计算【训练=tr+va】 和 【测试=te】上的 Accuracy 和 Kappa
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

def abGridSearchRes():
	# top5HyperParaList = [Base Learner Number, ...]
	top5HyperParaList = [350, 200, 250, 150, 300]
	
	# NH55 D:\cs_0\git_prjs\dpfc_large_files\ml_sl\ab\trained_on_tr_tested_on_vail
	modelFp = '../../../../dpfc_large_files/ml_sl/ab/trained_on_tr_tested_on_vail/'
	fnList = os.listdir(modelFp)
	
	for hyperPara in top5HyperParaList:
		targetFnList = []
		for fn in fnList:
			# 2020_04_23_ab_boost_num=150_0_pickle.file
			baseLearnerNumber = float(fn.split('_')[5].split('=')[1])
			if hyperPara == baseLearnerNumber:
				targetFnList.append(fn)
		
		abGSTrAccList = []
		abGSTrKappaList = []
		abGSVaAccList = []
		abGSVaKappaList = []
		for targetFn in targetFnList:
			modelAbsFp = os.path.join(modelFp, targetFn)
			"""
			base learner config in GS: Multiclassification LRC:
				(OvO, Linear α, Initial α = 1, Iteration = 9000) Number = 150
			"""
			abGSTr = AB(boost_num=int(hyperPara), resample_num=10, alpha_init=1, max_iter=9000,
						unlabeled_dataset_list=tr_data_list,
						labeled_dataset_list=tr_dataset,
						label_list=label_list)
			abGSTrSample_label_prob_dict_list = abGSTr.classify(ab_model_name=modelAbsFp)
			abGSTrAcc = cal_accuracy(abGSTrSample_label_prob_dict_list, tr_label_list)
			abGSTrAccList.append(abGSTrAcc)
			abGSTrKappa = cal_kappa(abGSTrSample_label_prob_dict_list, tr_label_list)
			abGSTrKappaList.append(abGSTrKappa)
			
			abGSTr.unlabeled_dataset_list = va_data_list
			abGSVaSample_label_prob_dict_list = abGSTr.classify(ab_model_name=modelAbsFp)
			abGSVaAcc = cal_accuracy(abGSVaSample_label_prob_dict_list, va_label_list)
			abGSVaAccList.append(abGSVaAcc)
			abGSVaKappa = cal_kappa(abGSVaSample_label_prob_dict_list, va_label_list)
			abGSVaKappaList.append(abGSVaKappa)
			print('GS: Base learner number={0} ==> Tr-Acc={1},Tr-Kappa={2}==>Va-Acc={3},Va-Kappa={4}'.format(
				hyperPara,
				abGSTrAcc,abGSTrKappa,abGSVaAcc,abGSVaKappa))
		print('GS: Base learner number={0} ==>Tr-Acc-Avg={1},Tr-Kappa-Avg={2}==>Va-Acc-Avg={3},Va-Kappa-Avg={4}'.format(
			hyperPara,
			sum(abGSTrAccList) / len(abGSTrAccList),
			sum(abGSTrKappaList) / len(abGSTrKappaList),
			sum(abGSVaAccList) / len(abGSVaAccList),
			sum(abGSVaKappaList) / len(abGSVaKappaList)
		))
# abGridSearchRes()
"""
GS: Base learner number=350 ==> Tr-Acc=0.48873873873873874,Tr-Kappa=0.32335703208394595==>Va-Acc=0.44680851063829785,Va-Kappa=0.2623000301841231
GS: Base learner number=350 ==> Tr-Acc=0.5247747747747747,Tr-Kappa=0.4133640581851881==>Va-Acc=0.43617021276595747,Va-Kappa=0.30670748677984977
GS: Base learner number=350 ==> Tr-Acc=0.5157657657657657,Tr-Kappa=0.38678760470733325==>Va-Acc=0.4574468085106383,Va-Kappa=0.3153384747215081
GS: Base learner number=350 ==> Tr-Acc=0.4797297297297297,Tr-Kappa=0.3621443452843683==>Va-Acc=0.46808510638297873,Va-Kappa=0.3465869595440011
GS: Base learner number=350 ==> Tr-Acc=0.5247747747747747,Tr-Kappa=0.4005029724005093==>Va-Acc=0.44680851063829785,Va-Kappa=0.28953488372093017
GS: Base learner number=350 ==> Tr-Acc=0.5247747747747747,Tr-Kappa=0.4005029724005093==>Va-Acc=0.44680851063829785,Va-Kappa=0.28953488372093017
GS: Base learner number=350 ==> Tr-Acc=0.47297297297297297,Tr-Kappa=0.3562226972766986==>Va-Acc=0.40425531914893614,Va-Kappa=0.26959900097127787
GS: Base learner number=350 ==> Tr-Acc=0.47297297297297297,Tr-Kappa=0.3562226972766986==>Va-Acc=0.40425531914893614,Va-Kappa=0.26959900097127787
GS: Base learner number=350 ==> Tr-Acc=0.4797297297297297,Tr-Kappa=0.3422095663216223==>Va-Acc=0.46808510638297873,Va-Kappa=0.333805811481219
GS: Base learner number=350 ==> Tr-Acc=0.4594594594594595,Tr-Kappa=0.34694678008481855==>Va-Acc=0.3404255319148936,Va-Kappa=0.21029810298102977
GS: Base learner number=350 ==> Tr-Acc=0.5292792792792793,Tr-Kappa=0.4052072249926289==>Va-Acc=0.48936170212765956,Va-Kappa=0.3548756076637117
GS: Base learner number=350 ==> Tr-Acc=0.48873873873873874,Tr-Kappa=0.34769692772682853==>Va-Acc=0.46808510638297873,Va-Kappa=0.32335156924848835
GS: Base learner number=350 ==> Tr-Acc-Avg=0.4968093093093094,Tr-Kappa-Avg=0.37009707322842916,AK=0.8669063825377386
							==> Va-Acc-Avg=0.43971631205673756,Va-Kappa-Avg=0.29762765099902894,AK=0.7373439630557665

GS: Base learner number=200 ==> Tr-Acc=0.5022522522522522,Tr-Kappa=0.3879452840898459==>Va-Acc=0.3829787234042553,Va-Kappa=0.23254504504504506
GS: Base learner number=200 ==> Tr-Acc=0.5180180180180181,Tr-Kappa=0.40011364353810214==>Va-Acc=0.39361702127659576,Va-Kappa=0.24118396827644809
GS: Base learner number=200 ==> Tr-Acc=0.42342342342342343,Tr-Kappa=0.28610189930660235==>Va-Acc=0.32978723404255317,Va-Kappa=0.1790962018297754
GS: Base learner number=200 ==> Tr-Acc=0.509009009009009,Tr-Kappa=0.3921322347266881==>Va-Acc=0.43617021276595747,Va-Kappa=0.2989023360540389
GS: Base learner number=200 ==> Tr-Acc=0.46396396396396394,Tr-Kappa=0.31291694300316==>Va-Acc=0.43617021276595747,Va-Kappa=0.2795372378886479
GS: Base learner number=200 ==> Tr-Acc=0.5540540540540541,Tr-Kappa=0.4447791735349287==>Va-Acc=0.43617021276595747,Va-Kappa=0.2879805630984708
GS: Base learner number=200 ==> Tr-Acc=0.5022522522522522,Tr-Kappa=0.3733979578152838==>Va-Acc=0.39361702127659576,Va-Kappa=0.23631698973774232
GS: Base learner number=200 ==> Tr-Acc=0.5157657657657657,Tr-Kappa=0.39774770511971225==>Va-Acc=0.44680851063829785,Va-Kappa=0.3063715056052221
GS: Base learner number=200 ==> Tr-Acc=0.5495495495495496,Tr-Kappa=0.4319235911641089==>Va-Acc=0.425531914893617,Va-Kappa=0.27016534867002157
GS: Base learner number=200 ==> Tr-Acc=0.527027027027027,Tr-Kappa=0.40127143132344434==>Va-Acc=0.4574468085106383,Va-Kappa=0.30642361111111116
GS: Base learner number=200 ==> Tr-Acc-Avg=0.5065315315315315,Tr-Kappa-Avg=0.38283298636218765,AK=0.8893645178937192
							==> Va-Acc-Avg=0.4138297872340426,Va-Kappa-Avg=0.2638522807316524,AK=0.677682067965695

GS: Base learner number=250 ==> Tr-Acc=0.509009009009009,Tr-Kappa=0.37483772210273403==>Va-Acc=0.4787234042553192,Va-Kappa=0.33275387512675647
GS: Base learner number=250 ==> Tr-Acc=0.509009009009009,Tr-Kappa=0.39101930905178656==>Va-Acc=0.3723404255319149,Va-Kappa=0.23174954979914117
GS: Base learner number=250 ==> Tr-Acc=0.5202702702702703,Tr-Kappa=0.3861910510533899==>Va-Acc=0.35106382978723405,Va-Kappa=0.19534100477126018
GS: Base learner number=250 ==> Tr-Acc=0.527027027027027,Tr-Kappa=0.3907714020621251==>Va-Acc=0.5319148936170213,Va-Kappa=0.3949678174370977
GS: Base learner number=250 ==> Tr-Acc=0.5135135135135135,Tr-Kappa=0.3750635336435078==>Va-Acc=0.3829787234042553,Va-Kappa=0.21259387637203928
GS: Base learner number=250 ==> Tr-Acc=0.47297297297297297,Tr-Kappa=0.3309679122690655==>Va-Acc=0.4787234042553192,Va-Kappa=0.33678905687544997
GS: Base learner number=250 ==> Tr-Acc=0.4864864864864865,Tr-Kappa=0.35049820031951545==>Va-Acc=0.40425531914893614,Va-Kappa=0.23466123873218953
GS: Base learner number=250 ==> Tr-Acc=0.5022522522522522,Tr-Kappa=0.37441265914785365==>Va-Acc=0.3723404255319149,Va-Kappa=0.20884450784593436
GS: Base learner number=250 ==> Tr-Acc=0.5022522522522522,Tr-Kappa=0.37592856415996745==>Va-Acc=0.425531914893617,Va-Kappa=0.27938671209540034
GS: Base learner number=250 ==> Tr-Acc=0.5022522522522522,Tr-Kappa=0.3830735666727442==>Va-Acc=0.3404255319148936,Va-Kappa=0.1915660979331391
GS: Base learner number=250 ==> Tr-Acc-Avg=0.5045045045045045,Tr-Kappa-Avg=0.373276392048269,AK=0.8777808965527735
							==> Va-Acc-Avg=0.4138297872340425,Va-Kappa-Avg=0.2618653736988408,AK=0.6756951609328833

GS: Base learner number=150 ==> Tr-Acc=0.5022522522522522,Tr-Kappa=0.3639834326123451==>Va-Acc=0.46808510638297873,Va-Kappa=0.3348429097084631
GS: Base learner number=150 ==> Tr-Acc=0.4436936936936937,Tr-Kappa=0.3198335369673214==>Va-Acc=0.2872340425531915,Va-Kappa=0.139382344902979
GS: Base learner number=150 ==> Tr-Acc=0.4864864864864865,Tr-Kappa=0.35265793159015485==>Va-Acc=0.40425531914893614,Va-Kappa=0.24767757610404453
GS: Base learner number=150 ==> Tr-Acc=0.5202702702702703,Tr-Kappa=0.3867562380038388==>Va-Acc=0.3723404255319149,Va-Kappa=0.19401249818340358
GS: Base learner number=150 ==> Tr-Acc=0.5202702702702703,Tr-Kappa=0.3828020988331115==>Va-Acc=0.4787234042553192,Va-Kappa=0.3231447465099192
GS: Base learner number=150 ==> Tr-Acc=0.5022522522522522,Tr-Kappa=0.37005912677267966==>Va-Acc=0.43617021276595747,Va-Kappa=0.28036978188646544
GS: Base learner number=150 ==> Tr-Acc=0.49774774774774777,Tr-Kappa=0.37799890691845234==>Va-Acc=0.40425531914893614,Va-Kappa=0.27071210861734546
GS: Base learner number=150 ==> Tr-Acc=0.5202702702702703,Tr-Kappa=0.40505039066923343==>Va-Acc=0.43617021276595747,Va-Kappa=0.29212844558113105
GS: Base learner number=150 ==> Tr-Acc=0.5157657657657657,Tr-Kappa=0.3924195652865735==>Va-Acc=0.3829787234042553,Va-Kappa=0.2089378990133488
GS: Base learner number=150 ==> Tr-Acc=0.46396396396396394,Tr-Kappa=0.32773494159859523==>Va-Acc=0.44680851063829785,Va-Kappa=0.29577870623829416
GS: Base learner number=150 ==> Tr-Acc-Avg=0.49729729729729727,Tr-Kappa-Avg=0.3679296169252305,AK=0.8652269142225277
							==> Va-Acc-Avg=0.4117021276595745,Va-Kappa-Avg=0.2586987016745394,AK=0.6704008293341139

GS: Base learner number=300 ==> Tr-Acc=0.509009009009009,Tr-Kappa=0.38602845561976296==>Va-Acc=0.39361702127659576,Va-Kappa=0.23751245197096912
GS: Base learner number=300 ==> Tr-Acc=0.4166666666666667,Tr-Kappa=0.2830315726470148==>Va-Acc=0.3723404255319149,Va-Kappa=0.23355444997236038
GS: Base learner number=300 ==> Tr-Acc=0.5563063063063063,Tr-Kappa=0.4471503605898377==>Va-Acc=0.425531914893617,Va-Kappa=0.2908633696563286
GS: Base learner number=300 ==> Tr-Acc=0.536036036036036,Tr-Kappa=0.43132134596732075==>Va-Acc=0.3617021276595745,Va-Kappa=0.22120961060480532
GS: Base learner number=300 ==> Tr-Acc=0.5180180180180181,Tr-Kappa=0.3818730646126624==>Va-Acc=0.4148936170212766,Va-Kappa=0.2659378105920772
GS: Base learner number=300 ==> Tr-Acc=0.5202702702702703,Tr-Kappa=0.3960071018917089==>Va-Acc=0.43617021276595747,Va-Kappa=0.27734261676820426
GS: Base learner number=300 ==> Tr-Acc=0.46621621621621623,Tr-Kappa=0.352753755120619==>Va-Acc=0.39361702127659576,Va-Kappa=0.25748337028824836
GS: Base learner number=300 ==> Tr-Acc=0.46621621621621623,Tr-Kappa=0.352753755120619==>Va-Acc=0.39361702127659576,Va-Kappa=0.25748337028824836
GS: Base learner number=300 ==> Tr-Acc=0.5112612612612613,Tr-Kappa=0.38292258031459747==>Va-Acc=0.425531914893617,Va-Kappa=0.27361190612478536
GS: Base learner number=300 ==> Tr-Acc=0.5112612612612613,Tr-Kappa=0.38292258031459747==>Va-Acc=0.425531914893617,Va-Kappa=0.27361190612478536
GS: Base learner number=300 ==> Tr-Acc=0.4797297297297297,Tr-Kappa=0.34496963194298086==>Va-Acc=0.4148936170212766,Va-Kappa=0.259948468365302
GS: Base learner number=300 ==> Tr-Acc=0.47297297297297297,Tr-Kappa=0.3512906005319747==>Va-Acc=0.425531914893617,Va-Kappa=0.27918205055381995
GS: Base learner number=300 ==> Tr-Acc-Avg=0.49699699699699695,Tr-Kappa-Avg=0.374418733722808,AK=0.8714157307198049
							==> Va-Acc-Avg=0.4069148936170213,Va-Kappa-Avg=0.26064511510916116,AK=0.6675600087261825
"""
	
def abFinalRes():
	finalModelFp = './models/trained_on_TV_tested_on_test/2020_07_06_ab_final_boost_num=150_3_pickle.file'
	
	ab = AB(boost_num=150, resample_num=10, alpha_init=1, max_iter=9000,
			unlabeled_dataset_list=tr_va_data_list,
			labeled_dataset_list=tr_va_dataset,
			label_list=label_list)
	
	# res on trVa
	abTrVaSample_label_prob_dict_list = ab.classify(ab_model_name=finalModelFp)
	abTrVaAcc = cal_accuracy(abTrVaSample_label_prob_dict_list, tr_va_label_list)
	abTrVaKappa = cal_kappa(abTrVaSample_label_prob_dict_list, tr_va_label_list)
	
	# res on te
	ab.unlabeled_dataset_list = te_data_list
	abTeSample_label_prob_dict_list = ab.classify(ab_model_name=finalModelFp)
	abTeAcc = cal_accuracy(abTeSample_label_prob_dict_list, te_label_list)
	abTeKappa = cal_kappa(abTeSample_label_prob_dict_list, te_label_list)
	
	print('AB final res: abTrVaAcc={0}, abTrVaKappa={1}, abTrVaAK={2}, abTeAcc={3}, abTeKappa={4}, abTeAK={5}'.format(
		abTrVaAcc, abTrVaKappa, abTrVaAcc+abTrVaKappa,
		abTeAcc, abTeKappa, abTeAcc+abTeKappa
	))
# abFinalRes()
"""
AB final res:
	abTrVaAcc=0.45353159851301117, 	abTrVaKappa=0.3205932760331431, 	abTrVaAK=0.7741248745461542,
	abTeAcc=0.5714285714285714, 	abTeKappa=0.4740663900414937, 		abTeAK=1.045494961470065
"""
