import os

from a_d.ny_AD import detect as nyImDetect
from a_d.absZ_AD_1 import detect as absZDetect
from a_d.phase_AD_1 import detect as phaseDetect
from data_processor.GOA_preprocessor.goa_data_wrapper import load_Lai_EIS_data
from data_processor.GOA_preprocessor.goa_data_wrapper import load_lai_manual_fitting_res
from IS.IS import IS_0
from goa.integration.goa_intergration import goa_fitter_1
from ml_sl.adaboost.ab_0 import AB
from ml_sl.ml_data_wrapper import split_labeled_dataset_list
from utils.file_utils.dataset_reader_pack.ml_dataset_reader import get_TV_T_dataset, get_T_V_T_dataset
from playground.laiZhaoGui.getLaiVogitAddC import getLaiVogitAddCResDict

"""
Module Function
    This module aims to guide you to use AIA-EIS to automatically parse an EIS.

Workflow
	ECM selection using AdaBoost
		an ECM prediction by AdaBoost
	ECM Parameter Identification
		Abormal points identification and removal by a newly proposed routine
		Use a global optimization algorithm to identify the parameters of the ECM
"""

# --------------------------------------------------------- ECM selection using AdaBoost ---------------------------------------------------------
label_list = [2, 4, 5, 6, 7, 8, 9]
# Import ml-dataset (Training, validation, Test)
ml_dataset_pickle_file_path = './datasets/ml_datasets/normed'
tr_dataset, va_dataset, te_dataset = get_T_V_T_dataset(file_path=ml_dataset_pickle_file_path)
ml_dataset = tr_dataset + va_dataset + te_dataset
ml_label_list, ml_data_list = split_labeled_dataset_list(ml_dataset)

# Use AdaBoost to predict an ECM
# You can pick any number between 0 ~ 620 (total data amount of ml-dataset 629)
eisNum = 4
ab = AB(boost_num=150, resample_num=10, alpha_init=1, max_iter=9000,
        unlabeled_dataset_list=[ml_data_list[eisNum]],
        labeled_dataset_list=tr_dataset,
        label_list=label_list)

abSampleLabelProbDictList = ab.classify(ab_model_name='./ml_sl/adaboost/models/trained_on_TV_tested_on_test/2020_07_06_ab_final_boost_num=150_3_pickle.file')
print('Probability of each ECM', abSampleLabelProbDictList) # the Probability is not normalized
# Probability of each ECM [{2: 31.98819325408522, 4: 202.87789814399957, 5: 171.18324221610857, 6: 178.12321996038756, 7: 97.85950318745802, 8: 100.41176386279655, 9: 14.88727029490577}] --> ECM4
# --------------------------------------------------------- ECM selection using AdaBoost ---------------------------------------------------------

# ---------------------------------- ECM parameter identification using GOA ----------------------------------
# Load an piece of experiment EIS
lai_normed_eis_dict_list = load_Lai_EIS_data(file_path='./datasets/goa_datasets/normed',
                                             file_name='2020_08_22_goa_lai_normed_dataset_pickle.file')
lai_manual_fit_res_dict = load_lai_manual_fitting_res(file_path='./datasets/goa_datasets/Lai_manual_fitting_res',
                                                      file_names=['2020_07_22_lai_ecm2_fitting_res.CSV',
                                                                  '2020_07_22_lai_ecm9_fitting_res.CSV'])
laiVogitAddCResDict = getLaiVogitAddCResDict(fp='./playground/laiZhaoGui/', fn='laiAddVogitCRes.txt')

# You can pick any number between 0 ~ 100 (total data amount of goa-real-dataset around 110)
eisNum1 = 15
eisDict = {}
eisDict['exp_fn'] = lai_normed_eis_dict_list[eisNum1]['file_name']
eisDict['ecm_num'] = lai_normed_eis_dict_list[eisNum1]['ecm_num']
eisDict['limit'] = lai_manual_fit_res_dict[eisDict['exp_fn']]['limit']

normedRawEIS = IS_0()
normedRawEIS.readFromLaiPickle(laiNormedEisDict=lai_normed_eis_dict_list[eisNum1], limitList=None)
deletedPointIndex_list = nyImDetect(eis_source=normedRawEIS, vogitAddC=laiVogitAddCResDict[eisDict['exp_fn']],
                                    pointNum=10, chiSquareLimit=2.5 * 1e-2, printFlag=False)
for dpi in deletedPointIndex_list:
    normedRawEIS.removeZByIndex(index=dpi)

eisDict['f'] = normedRawEIS.fre_arr.tolist()
eisDict['z_raw'] = normedRawEIS.z_arr.tolist()

goa_fitter_1(ecm_para_config_dict=eisDict, repeat_time=1)
"""
The fitted parameters will be saved into a txt file in the root, like
    1-20_AD_Fit_Res.txt
        ECM-Num,GOA-Name,Repeat-Time,Iteration,Fitted-Parameters-List,Chi-Square
        9,WOA,0,5000,[8.209658683025127e-18,0.0002799766946517235,0.8309902623933234,36.90995369771564,0.00021694289934531976,0.6667090047271117,89623.59962536128],0.006657020048254013
        9,DE,0,5000,[5.519960373820262e-19,0.0002320087447699836,0.8466398593116743,19.083098254444906,0.00023118964439438716,0.7396497513914675,14618.120195175085],0.003906243712699124
        9,ABC,0,5000,[9.05996570343743e-18,0.0002473084669576885,0.8415734476361887,21.78126015950854,0.00020424466012936251,0.7513557092899907,14584.264829355418],0.004126290671121253
        9,GSO,0,70,[7.539878307247104e-18,0.0004004728075715407,0.8001020501304705,691.5902899740187,0.0001030997630154698,0.6038230328289205,52031.185468292766],0.011474664563568729
"""
# ---------------------------------- ECM parameter identification using GOA ----------------------------------