import os

# 1- Parse each file, like '1-1_fitted_res_ecmNum=9.txt', to get the best fitting result for that file, '1-1'
def get_GOAs_best_fitting_res(fp, order_str='2nd'):
    """
    Function
        解析每种ECM对应的Top5 的GOA的结果
    :param
        fp:
        order_str: str
            '2nd'：改正了Chi-Square的计算
            '1st'：计算Chi-Square的时候没有 除以 自由度v
    :return:
    """
    goa_fit_res_dict_list = []
    fn_list = os.listdir(fp)
    for fn in fn_list:
        if fn.endswith('.txt'):
            # read this file
            with open(os.path.join(fp, fn), 'r') as file:
                goa_fit_res_dict = {}
                if order_str == '1st':
                    # file name: 1-1_fitted_res_ecmNum=9.txt
                    goa_fit_res_dict['fn'] = fn.split('.')[0].split('_')[0]
                    goa_fit_res_dict['ecm_num'] = int(fn.split('.')[0].split('=')[1])
                elif order_str == '2nd':
                    # file name: 2021_02_09_1-1_fitted_res_ecmNum=9.txt
                    goa_fit_res_dict['fn'] = fn.split('.')[0].split('_')[3]
                    goa_fit_res_dict['ecm_num'] = int(fn.split('.')[0].split('=')[1])

                # the first 3 lines in a file looks like the following:
                # ECM-Num,GOA-Name,Repeat-Time,Iteration,Fitted-Parameters-List,Chi-Square
                # 9,DE,0,10000,[0.0034491063702477235,0.0005193591443538511,0.8993090766325862,49.803159866657595,0.0013381630590298477,0.5737308787670669,99999.99999999933],0.023370590793507506
                # 9,DE,1,10000,[0.0034491063657608634,0.0005193591433264387,0.8993090768295705,49.80315900777707,0.0013381630582704103,0.5737308771104276,100000.0],0.02337059079350752
                chi_s_list = []
                line_list = []
                for line in file.readlines()[1:]:
                    line_list.append(line)
                    line_str_list = line.strip().split(',')
                    chi_s_list.append(float(line_str_list[-1]))
                min_chi_s_index = chi_s_list.index(min(chi_s_list))

                # IndexError: list index out of range, file.readlines() returns a generator for only one time
                optimal_res_line = line_list[min_chi_s_index]
                res_str_list = optimal_res_line.strip().split(',')
                goa_fit_res_dict['goa_name'] = res_str_list[1]
                # goa_fit_res_dict['iter'] = int(res_str_list[2])
                goa_fit_res_dict['para'] = eval(','.join(res_str_list[4 : len(res_str_list) - 1]))
                goa_fit_res_dict['chi_square'] = float(res_str_list[-1])

                goa_fit_res_dict_list.append(goa_fit_res_dict)
    return goa_fit_res_dict_list

# 2- Match coordinate for each file
def pack_GOAs_fit_res(goa_fit_res_dict_list, coor_dict_list):
    for goa_fit_res_dict in goa_fit_res_dict_list:
        fn = goa_fit_res_dict['fn']
        for coor_dict in coor_dict_list:
            if fn == coor_dict['fn']:
                goa_fit_res_dict['coor'] = coor_dict['coor']
                break
    return goa_fit_res_dict_list

def wrap_GOAs_data_4_contour(goa_fit_res_dict_list):
    x_list = []
    y_list = []
    z_list = []
    for goa_fit_res_dict in goa_fit_res_dict_list:
        x, y = goa_fit_res_dict['coor']
        z = goa_fit_res_dict['chi_square']
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
    return x_list, y_list, z_list