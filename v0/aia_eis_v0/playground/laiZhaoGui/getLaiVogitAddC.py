import os
from IS.IS import IS_0
from data_processor.GOA_preprocessor.goa_data_wrapper import load_Lai_EIS_data
from utils.visualize_utils.IS_plots.ny import nyquist_plot

"""
Function
    画出LaiEIS所有的Nyquist，肉眼判断Vogit加不加C
"""

def getLaiVogitAddCRes():
    """
    lai_normed_eis_dict_list[
        dict0{
            'file_name': '1-1',
            'ecm_num': 9,
            'f': [100078.1, 63140.62, ..., 0.1588983, 0.1001603],
            'z_raw': [(0.005566658429999999-0.0112022736j),
                      (0.006214947129999999-0.0172324988j),
                      ...,
                      (285.52881799999994-486.4391289999999j),
                      (370.64242699999994-661.259928j)]
        }
        dict1,
        ...
    ]
    """
    lai_normed_eis_dict_list = load_Lai_EIS_data(file_path='../../datasets/goa_datasets/normed',
                                                 file_name='2020_08_22_goa_lai_normed_dataset_pickle.file')
    normedRawEIS = IS_0()
    for lai_normed_eis_dict in lai_normed_eis_dict_list:
        print('FileName', lai_normed_eis_dict['file_name'])
        normedRawEIS.readFromLaiPickle(laiNormedEisDict=lai_normed_eis_dict, limitList=None)
        nyquist_plot(z_list=normedRawEIS.z_arr,
                     grid_flag=False, fig_title='Nyquist of {}'.format(lai_normed_eis_dict['file_name']))
# getLaiVogitAddCRes()

def getLaiVogitAddCResDict(fp, fn):
    """

    :param fp:
    :param fn:
    :return:

    laiVogitAddCResDict {
        '1-1':True == Add C to Vogit,
        '1-2':False == No C to Vogit
    }
    """
    laiVogitAddCResDict = {}
    with open(os.path.join(fp, fn), 'r') as f:
        for line in f.readlines()[1:]:
            fileName, boolStr = line.strip().split(',')
            if boolStr == 'Y':
                laiVogitAddCResDict[fileName] = True
            elif boolStr == 'N':
                laiVogitAddCResDict[fileName] = False
    return laiVogitAddCResDict