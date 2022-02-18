import xlrd

from utils.file_utils.filename_utils import get_date_prefix

class goa_scorer():
    """
    Function
        1- 对GOA的【Averaged Chi-Squared】打分
            1.1- Collect all GOAs' 【Averaged Chi-Squared】 on ECM-1
            1.2- 按照10分制为每个GOA在ECM-1的Averaged Chi-Squared这一标准上打分
            1.3- 十分位以后的结果 打 0分最差；最靠前的结果打10分 最好
        2- 在GOA的【Averaged Running Time】和【RMSE of Chi-Squared】两个指标上评分，流程同上
    """
    def __init__(self, excel_path, sheet_name):
        """
        :param
            goa_perf_dict{
                            'DE':{
                                    1: [0(avg_chi_s), 10.18641096(avg_t), 0(rmse)]
                                    2: [0.035368762	127.7086203	0.190466611]
                                    ...
                                    9: [0.045966172	109.7103808	0.13241534]
                                 },
                            'EDA':{
                                    1: [0(avg_chi_s), 10.18641096(avg_t), 0(rmse)]
                                    2: [0.035368762	127.7086203	0.190466611]
                                    ...
                                    9: [0.045966172	109.7103808	0.13241534]
                                 },
                            ...
                            'GA'
                         }
            ecm_perf_dict{
                            1:{
                                'avg_Chi_Square': [(0, 'DE'), (2.151850743, 'EDA'), (5.273726272, 'EP'), ...],
                                'avg_t': [(10.18641096, 'DE'), (105.1849383, 'EDA'), (321.6677206, 'EP'), ...],
                                'rmse': [(0, 'DE'), (1.332567623, 'EDA'), (2.824279146, 'EP'), ...],
                              }
                         }
            ecm_score_range_dict{
                                    1:{
                                        'avg_Chi_Square': [第1个十分位， 第2个十分位, ..., 第8个十分位，第9个十分位],
                                        'avg_t': [第1个十分位， 第2个十分位, ..., 第8个十分位，第9个十分位],
                                        'rmse': [第1个十分位， 第2个十分位, ..., 第8个十分位，第9个十分位]
                                      }
                                }
        """
        self.goa_perf_dict = self.read_excel(excel_path, sheet_name)
        self.ecm_perf_dict = self.group_by_data_ecmNum()
        self.ecm_score_range_dict = self.get_ecm_score_range()

    def read_excel(self, excel_path, sheet_name):
        workbook = xlrd.open_workbook(excel_path)
        sheet = workbook.sheet_by_name(sheet_name)
        # For testing
        for i in range(10):
            print(i, sheet.cell(rowx=i, colx=i))

        goa_perf_dict = {}
        # data starts from row 5 and col 3
        for r in range(4, sheet.nrows):
            # access whether the row is empty by detect the last cell in a row is empty or not
            if sheet.cell(rowx=r, colx=6).value == '':
                break

            first_str = sheet.cell(rowx=r, colx=2).value
            if first_str != '':
                goa_name = first_str
                goa_perf_dict[goa_name] = {}

            ecm_num = int(sheet.cell(rowx=r, colx=3).value)
            # avg_chi_s, avg_t, rmse = list(map(float, [sheet.cell(rowx=r, colx=i) for i in range(5, 8)]))
            goa_perf_dict[goa_name][ecm_num] = list(map(float, [sheet.cell(rowx=r, colx=i).value for i in range(4, 7)]))
        return goa_perf_dict

    def group_by_data_ecmNum(self):
        ecm_perf_dict = {}
        for goa_name, v in self.goa_perf_dict.items():
            for ecm_num, perf_list in v.items():
                if ecm_num not in ecm_perf_dict.keys():
                    ecm_perf_dict[ecm_num] = {}
                    ecm_perf_dict[ecm_num]['avg_Chi_Square'] = [(perf_list[0], goa_name)]
                    ecm_perf_dict[ecm_num]['avg_t'] = [(perf_list[1], goa_name)]
                    ecm_perf_dict[ecm_num]['rmse'] = [(perf_list[2], goa_name)]
                else:
                    ecm_perf_dict[ecm_num]['avg_Chi_Square'].append((perf_list[0], goa_name))
                    ecm_perf_dict[ecm_num]['avg_t'].append((perf_list[1], goa_name))
                    ecm_perf_dict[ecm_num]['rmse'].append((perf_list[2], goa_name))
        return ecm_perf_dict

    def get_ecm_score_range(self, output_sorted_scores_flag=False):
        ecm_score_range_dict = {}
        ecm_sorted_three_scores_dict = {}
        for ecm_num, one_ecm_perf_dict in self.ecm_perf_dict.items():
            if ecm_num not in ecm_score_range_dict.keys():
                ecm_score_range_dict[ecm_num] = {}
                ecm_sorted_three_scores_dict[ecm_num] = {}

            chi_square_sorted_list = sorted(one_ecm_perf_dict['avg_Chi_Square'], key=lambda pair: pair[0], reverse=False)
            t_sorted_list = sorted(one_ecm_perf_dict['avg_t'], key=lambda pair: pair[0], reverse=False)
            rmse_sorted_list = sorted(one_ecm_perf_dict['rmse'], key=lambda pair: pair[0], reverse=False)

            ecm_sorted_three_scores_dict[ecm_num]['avg_Chi_Square'] = chi_square_sorted_list
            ecm_sorted_three_scores_dict[ecm_num]['avg_t'] = t_sorted_list
            ecm_sorted_three_scores_dict[ecm_num]['rmse'] = rmse_sorted_list

            # Split the list length into 10 pieces
            step_size = int(len(chi_square_sorted_list) / 10)
            chi_square_10th_list = []
            t_10th_list = []
            rmse_square_10th_list = []

            # 将chi_square的min ~ max中间插入9个值，分成10份，不能进行0~10（11个数）打分
            # for i in range(step_size, len(chi_square_sorted_list), step_size):
            #     cs_10th = 0.5 * (chi_square_sorted_list[i - 1][0] + chi_square_sorted_list[i + 1][0])
            #     t_10th = 0.5 * (t_sorted_list[i - 1][0] + t_sorted_list[i + 1][0])
            #     rmse_10th = 0.5 * (rmse_sorted_list[i - 1][0] + rmse_sorted_list[i + 1][0])
            for i in range(0, len(chi_square_sorted_list), step_size):
                cs_10th = 0.5 * (chi_square_sorted_list[i][0] + chi_square_sorted_list[i + 1][0])
                t_10th = 0.5 * (t_sorted_list[i][0] + t_sorted_list[i + 1][0])
                rmse_10th = 0.5 * (rmse_sorted_list[i][0] + rmse_sorted_list[i][0])

                chi_square_10th_list.append(cs_10th)
                t_10th_list.append(t_10th)
                rmse_square_10th_list.append(rmse_10th)

            ecm_score_range_dict[ecm_num]['avg_Chi_Square'] = chi_square_10th_list
            ecm_score_range_dict[ecm_num]['avg_t'] = t_10th_list
            ecm_score_range_dict[ecm_num]['rmse'] = rmse_square_10th_list

        if output_sorted_scores_flag:
            return ecm_score_range_dict, ecm_sorted_three_scores_dict
        else:
            return ecm_score_range_dict

    def get_goa_scores(self):
        """
        :param
            goa_score_on_three_criterions_dict
            {
                ‘DE’:
                    {
                        1:[10(accuracy score), 10(running time score), 10(RMSE of Chi-Square score)]
                        2:[9, 10, 8],
                        ...
                        8:[],
                        9:[]
                    }
                'EDA':
                    {
                        1:[10(accuracy score), 10(running time score), 10(RMSE of Chi-Square score)]
                        2:[9, 10, 8],
                        ...
                        8:[],
                        9:[]
                    }
            }
            goa_weighted_score_dict
            {
                ‘DE’:
                    {
                        1: 10(weighted score) = 0.5 * 10(accuracy score) + 0.2 *  10(running time score) + 0.3 * 10(RMSE of Chi-Square score)]
                        2: 8.9,
                        ...
                        8: 7.2,
                        9: 9.4
                    }
                'EDA':
                    {
                        1: 4.5
                        2: 5.3,
                        ...
                        8: 3.6,
                        9: 3.8
                    }
            }
        """
        goa_score_on_three_criterions_dict = {}
        goa_weighted_score_dict = {}
        for goa_name, one_goa_perf_dict in self.goa_perf_dict.items():
            if goa_name not in goa_score_on_three_criterions_dict.keys():
                goa_score_on_three_criterions_dict[goa_name] = {}
                goa_weighted_score_dict[goa_name] = {}

            for ecm_num, perf_list in one_goa_perf_dict.items():
                avg_c_s, avg_t, rmse = perf_list
                chi_square_10th_list = self.ecm_score_range_dict[ecm_num]['avg_Chi_Square']
                t_10th_list = self.ecm_score_range_dict[ecm_num]['avg_t']
                rmse_square_10th_list = self.ecm_score_range_dict[ecm_num]['rmse']

                cs_score = None
                t_score = None
                rmse_score = None
                for i, c_s in enumerate(chi_square_10th_list):
                    if avg_c_s > chi_square_10th_list[-1]:
                        cs_score = 0
                        break
                    elif avg_c_s <= c_s:
                        cs_score = 10 - i
                        break
                    # if (i == len(chi_square_10th_list)) and (cs_score == None):
                    #     cs_score = 0
                for i, t in enumerate(t_10th_list):
                    if avg_t > t_10th_list[-1]:
                        t_score = 0
                        break
                    elif avg_t <= t:
                        t_score = 10 - i
                        break
                    # if (i == len(t_10th_list)) and (t_score == None):
                    #     t_score = 0
                for i, r in enumerate(rmse_square_10th_list):
                    if rmse >= rmse_square_10th_list[-1]:
                        rmse_score = 0
                        break
                    elif rmse <= r:
                        rmse_score = 10 - i
                        break
                    # if (i == len(rmse_square_10th_list)) and (rmse_score == None):
                    #     rmse_score = 0
                goa_score_on_three_criterions_dict[goa_name][ecm_num] = [cs_score, t_score, rmse_score]
                try:
                    goa_weighted_score_dict[goa_name][ecm_num] = 0.5 * cs_score + 0.3 * rmse_score + 0.2 * t_score
                except TypeError as e:
                    print(e)
        return goa_score_on_three_criterions_dict, goa_weighted_score_dict

def output_goa_three_scores_10th(ecm_score_range_dict, ecm_sorted_three_scores_dict):
    """
    Function
        输出 每种ECM上 20个GOA在 Avg-Chi-Square， Avg-Running-Time， 和 RMSE从小到大排序的20个结果
        like:
        1,
        chi_square:0.0,0.0,7.76209313310415e-17, ...,0.0512369777507534,0.0693911351583931,0.524574612867737 ,20个结果
        t:10.1864109577777,31.0974940855555, ...,382.840986806666,29636.0558249307 ,20个结果
        rmse:0.0,0.0,7.31750127352841e-16, ...,0.064086931761261,0.113304353969948 ,20个结果
        2, ...
    :param
        ecm_score_range_dict:
        ecm_sorted_three_scores_dict:
    :return:
    """
    three_score_fn = get_date_prefix() + 'goa_three_score.txt'
    with open(three_score_fn, 'a+') as file0:
        for ecm_num, data_dict in ecm_sorted_three_scores_dict.items():
            chi_square_sorted_list = data_dict['avg_Chi_Square']
            t_sorted_list = data_dict['avg_t']
            rmse_sorted_list = data_dict['rmse']
            line = str(ecm_num)+',\n'
            line += 'chi_square:'+','.join(list(map(str, [cs[0] for cs in chi_square_sorted_list])))+'\n'
            line += 't:'+','.join(list(map(str, [t[0] for t in t_sorted_list]))) + '\n'
            line += 'rmse:'+','.join(list(map(str, [rmse[0] for rmse in rmse_sorted_list]))) + '\n'
            file0.write(line)

    three_score_10th_fn = get_date_prefix() + 'goa_three_score_10th.txt'
    with open(three_score_10th_fn, 'a+') as file1:
        for ecm_num, score_10th_dict in ecm_score_range_dict.items():
            chi_square_10th_list = score_10th_dict['avg_Chi_Square']
            t_10th_list = score_10th_dict['avg_t']
            rmse_square_10th_list = score_10th_dict['rmse']
            line = str(ecm_num)+',\n'
            line += 'chi_square_10th:' + ','.join(list(map(str, [cs for cs in chi_square_10th_list]))) + '\n'
            line += 't_10th:' + ','.join(list(map(str, [t for t in t_10th_list]))) + '\n'
            line += 'rmse_10th:' + ','.join(list(map(str, [rmse for rmse in rmse_square_10th_list]))) + '\n'
            file1.write(line)

def output_goa_weighted_score(goa_weighted_score_dict):
    file_name = get_date_prefix() + 'goa_weighted_score.txt'
    with open(file_name, 'a+') as file:
        for k, v in goa_weighted_score_dict.items():
            # goa_weighted_score_dict['CSS']:
            #   {1: 3.5, 2: 1.5999999999999999, 3: 0.7,
            #   4: 6.8999999999999995, 5: 6.3, 7: 1.3, 8: 6.3}, 没有key=9的情况
            line_str = ','.join([k] + [str(v[i]) for i in range(1, 10)]) + '\n'
            file.write(line_str)

def output_goa_scores_on_three_criterions(goa_scores_on_three_criterions_dict):
    """
    Function
        每种GOA 在 每种ECM上的三个指标的分数（10分制）
        like:
            DE:10,10,10;8,10,7;3,6,2;6,10,3;8,5,7;9,3,6;9,10,7;7,10,3;9,10,7;
            EDA:2,8,3;3,8,4;2,6,4;1,6,2;2,9,2;2,8,2;2,7,5;1,6,3;2,6,2;
            EP:1,1,1;2,2,0;6,2,8;7,2,7;9,2,9;4,2,4;7,2,9;3,3,0;4,2,3;
            ...
    :param goa_scores_on_three_criterions_dict:
    :return:
    """
    file_name = get_date_prefix() + 'goa_scores_on_three_criterions.txt'
    with open(file_name, 'a+') as file:
        for goa_name, one_goa_scores_on_three_criterions_dict in goa_scores_on_three_criterions_dict.items():
            line_str = goa_name + ':'
            # for ecm_num, scores_list in one_goa_scores_on_three_criterions_dict.items():
                # 1:[10, 9, 6] ==> 1-10,9,6
                # line_str += str(ecm_num) + '-' + ','.join([str(s) for s in scores_list])+';'
            for k in sorted(one_goa_scores_on_three_criterions_dict.keys()):
                scores_list = one_goa_scores_on_three_criterions_dict[k]
                line_str += ','.join([str(s) for s in scores_list])+';'
            line_str += '\n'
            file.write(line_str)

# g_s = goa_scorer(excel_path='../../goa/goa_training_records.xlsx', sheet_name='goa_training_res_2')

# ecm_score_range_dict, ecm_sorted_three_scores_dict = g_s.get_ecm_score_range(output_sorted_scores_flag=True)
# output_goa_three_scores_10th(ecm_score_range_dict, ecm_sorted_three_scores_dict)

# goa_scores_on_three_criterions_dict, goa_weighted_score_dict = g_s.get_goa_scores()
# output_goa_weighted_score(goa_weighted_score_dict)
# output_goa_scores_on_three_criterions(goa_scores_on_three_criterions_dict)

"""
Generated R(RC)_IS_lin-kk_res.txt files:
    ----------------------------------------
    Third(Right)
        1- goa在time上的打分为何没有获得10分的goa？
            已解决：
                原来：‘avg_t < t’
                    avg_t和t经常一样，应设置为‘avg_t <= t’
        2- 这里的Chi-Square是正确的，除以v（自由度v = 测试点数N - ECM待拟合参数数量M）
        stored at dpfc_src\goa\simECM_res\3nd
        2021_03_07_goa_scores_on_three_criterions.txt
        2021_03_07_goa_three_score.txt
        2021_03_07_goa_three_score_10th.txt
        2021_03_07_goa_weighted_score.txt
    ----------------------------------------
    Second(Wrong)
        这里的Chi-Square是正确的，除以v（自由度v = 测试点数N - ECM待拟合参数数量M）
        stored at dpfc_src\goa\simECM_res\2nd
        2021_02_07_goa_scores_on_three_criterions.txt
        2021_02_07_goa_three_score.txt
        2021_02_07_goa_three_score_10th.txt
        2021_02_07_goa_weighted_score.txt
    ----------------------------------------
    First(Wrong)
        这里的Chi-Square是不正确的，没有除以v（自由度v = 测试点数N - ECM待拟合参数数量M）
        stored at dpfc_src\goa\simECM_res\1st
        2020_08_14_goa_weighted_score.txt
        2020_08_17_goa_scores_on_three_criterions.txt
        2020_09_30_goa_three_score.txt
        2020_09_30_goa_three_score_10th.txt
        2020_10_02_goa_scores_on_three_criterions.txt
        2020_10_02_goa_weighted_score.txt
"""