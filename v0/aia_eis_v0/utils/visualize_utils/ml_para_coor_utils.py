import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as p_go

"""
Parallel coordinate
    Draw by plotly
    refer:
        webs:
            Parallel Coordinates plot with Plotly Express
            https://plotly.com/python/parallel-coordinates-plot/
    plotly.express.parallel_coordinates
        color
        labels
            {strA : strA1,
             strB : strB1,
            }
            strA and strB are column names in DataFrame
            strA1 and strB1 are vertical axis names in parallel_coordinates
        color_continuous_scale
        color_continuous_midpoint

    Basic routine for DPFC to draw a Parallel coordinate:
        1- Convert the averaged grid search results txt file into a DataFrame object, df
        2- Input df into px.parallel_coordinates, set the following parameters:
            color
            labels
            color_continuous_scale
            color_continuous_midpoint
            
    Following Modification:
        Modify the color and value range of left side colorbar
"""
vmin, vmax = 0.0, 1.1

def svm_parallel_coor_3var_0(txt_file_path, column_names_list, colorBarMiddlePoint=0.25):
    """
    Function
        Draw parallel coordinates for 3 variables + 1 target, 4 columns
    :param txt_file_path:
    Attention:
        The step factor for parameters C and sigma are 10 times, like C: 0.1, 1, 10 and Sigma : 10, 100, 1000
        If directly plot parallel coordinates using raw data, it looks like svm_ovo_rbf_para_coor_0.png,
        In the C or sigma columns, the data are squeezed in the bottom part

        Convert column C and Sigma into log form log(C) and log(Sigma)
    version:
        0:
            The range of an axis is fixed as the minimum ~ maximum of this axis, can not changed.
            It is not convenient to compare with other parallel coordinates plots with different AK

            Plot function:
                plotly.express.parallel_coordinates
    """
    df = pd.read_table(txt_file_path, sep=',', header=None)
    # Give names to each column in df
    df.columns = ['Iteration', 'C', 'Sigma', 'AK']

    # logarithmetics the C and sigma columns
    df['C'] = df['C'].map(lambda x : np.log10(x))
    df['Sigma'] = df['Sigma'].map(lambda x : np.log10(x))
    # Modify Column names
    df.columns = column_names_list
    fig = px.parallel_coordinates(df, color=column_names_list[-1],
                                  labels={column_names_list[0]: column_names_list[0], column_names_list[1]: column_names_list[1],
                                          column_names_list[2]: column_names_list[2], column_names_list[3]: column_names_list[3]},
                                  color_continuous_scale = px.colors.diverging.Tealrose,
                                  color_continuous_midpoint = colorBarMiddlePoint)
    fig.show()
# ---------------------- Draw for SVM-OvO-RBF (3 variables)----------------------
# column_names_list = ['Iteration', 'Log10(C)', 'Log10(Sigma)', 'AK']
# svm_parallel_coor_3var_0(txt_file_path='../../ml_sl/svm/ovo_txt_res/2020_06_25_svm_ovo_rbf_gs_avg_res.txt',
#                        column_names_list=column_names_list, colorBarMiddlePoint=0.25)
# ---------------------- Draw for SVM-OvO-RBF (3 variables)----------------------

# ---------------------- Draw for SVM-OvR-RBF (3 variables)----------------------
# column_names_list = ['Iteration', 'Log10(C)', 'Log10(Sigma)', 'AK']
# svm_parallel_coor_3var_0(txt_file_path = '../../ml_sl/svm/ovr_txt_res/2020_06_25_svm_ovr_rbf_gs_avg_res.txt',
#                        column_names_list=column_names_list, colorBarMiddlePoint = 0.11)
# ---------------------- Draw for SVM-OvR-RBF (3 variables)----------------------

# ---------------------- Draw for SVM-OvO-Poly(2) (3 variables)----------------------
# column_names_list = ['Iteration', 'Log10(C)', 'Log10(Quadratic Coefficient)', 'AK']
# svm_parallel_coor_3var_0(txt_file_path='../../ml_sl/svm/ovo_txt_res/2020_06_25_svm_ovo_poly_gs_avg_res.txt',
#                        column_names_list=column_names_list, colorBarMiddlePoint=0.25)
# ---------------------- Draw for SVM-OvO-Poly(2) (3 variables)----------------------

# ---------------------- Draw for SVM-OvR-Poly(2) (3 variables)----------------------
# column_names_list = ['Iteration', 'Log10(C)', 'Log10(Quadratic Coefficient)', 'AK']
# svm_parallel_coor_3var_0(txt_file_path='../../ml_sl/svm/ovr_txt_res/2020_06_25_svm_ovr_poly_gs_avg_res.txt',
#                        column_names_list=column_names_list, colorBarMiddlePoint=0.7)
# ---------------------- Draw for SVM-OvR-Poly(2) (3 variables)----------------------

def svm_parallel_coor_3var_1(txt_file_path, column_names_list):
    """
    Function
        Draw parallel coordinates for 3 variables + 1 target, 4 columns
    :param txt_file_path:
    Attention:
        The step factor for parameters C and sigma are 10 times, like C: 0.1, 1, 10 and Sigma : 10, 100, 1000
        If directly plot parallel coordinates using raw data, it looks like svm_ovo_rbf_para_coor_0.png,
        In the C or sigma columns, the data are squeezed in the bottom part

        Convert column C and Sigma into log form log(C) and log(Sigma)
    version:
        0:
            The range of an axis is fixed as the minimum ~ maximum of this axis, can not changed.
            It is not convenient to compare with other parallel coordinates plots with different AK

            Plot function:
                plotly.express.parallel_coordinates

        1:
            Limit the range of certain axis

            Plot function:
                plotly.graph_objects.Figure
    """
    df = pd.read_table(txt_file_path, sep=',', header=None)
    # Give names to each column in df
    df.columns = ['Iteration', 'C', 'Sigma', 'AK']

    # logarithmetics the C and sigma columns
    df['C'] = df['C'].map(lambda x : np.log10(x))
    df['Sigma'] = df['Sigma'].map(lambda x : np.log10(x))
    # Modify Column names
    df.columns = column_names_list

    # ---------------------------------- Draw parallel coordinates in version 0 ----------------------------------
    # fig = px.parallel_coordinates(df, color=column_names_list[-1],
    #                               labels={column_names_list[0]: column_names_list[0], column_names_list[1]: column_names_list[1],
    #                                       column_names_list[2]: column_names_list[2], column_names_list[3]: column_names_list[3]},
    #                               color_continuous_scale = px.colors.diverging.Tealrose,
    #                               color_continuous_midpoint = colorBarMiddlePoint)
    # fig.show()
    # ---------------------------------- Draw parallel coordinates in version 0 ----------------------------------

    # ---------------------------------- Draw parallel coordinates in version 1 ----------------------------------
    # `colorscale` may be a palette name string of the following list: Greys, YlGnBu, Greens, YlOrRd, Bluered, RdBu, Reds,
    # Blues, Picnic, Rainbow, Portland, Jet, Hot, Blackbody, Earth, Electric, Viridis, Cividis
    # Colorscale possible candidates: Bluered, Picnic, Rainbow, Portland, Jet,
    fig = p_go.Figure(data = p_go.Parcoords(line = dict(color = df[column_names_list[-1]],
                                                        colorscale = 'Jet',
                                                        showscale = True,
                                                        cmin = vmin,
                                                        cmax = vmax
                                                        ),
                                            dimensions = list([dict(range=[df[column_names_list[0]].min(), df[column_names_list[0]].max()],
                                                                    label=column_names_list[0],
                                                                    values=df[column_names_list[0]]),
                                                               dict(range=[df[column_names_list[1]].min(), df[column_names_list[1]].max()],
                                                                    label=column_names_list[1],
                                                                    values=df[column_names_list[1]]),
                                                               dict(range=[df[column_names_list[2]].min(), df[column_names_list[2]].max()],
                                                                    label=column_names_list[2],
                                                                    values=df[column_names_list[2]]),
                                                               dict(range=[vmin, vmax],
                                                                    label=column_names_list[3],
                                                                    values=df[column_names_list[3]])])
                                            )
                      )
    # ----------- Set fonts and labels -----------
    fig.update_layout(
        font_family="Times New Roman",
        font_color="black",
        font_size=20,
        # title_font_family="Times New Roman",
        # title_font_color="red",
        legend_title_font_color="black"
    )
    # ----------- Set fonts and labels -----------
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    fig.show()
# ------------------------------------------- Draw parallel coordinates in version 1 -------------------------------------------

# ---------------------- Draw for SVM-OvO-RBF (3 variables)----------------------
# column_names_list = ['Iteration', 'Log10(C)', 'Log10(Sigma)', 'AK']
# column_names_list = ['Iteration', 'log{0}(C)'.format('\u2081\u2080'), 'log{0}(σ)'.format('\u2081\u2080'), 'AK']
# svm_parallel_coor_3var_1(txt_file_path='../../ml_sl/svm/ovo_txt_res/trained_on_tr_tested_on_vali/avg_res_and_plots/2020_06_25_svm_ovo_rbf_gs_avg_res.txt',
#                          column_names_list=column_names_list)
# ---------------------- Draw for SVM-OvO-RBF (3 variables)----------------------

# ---------------------- Draw for SVM-OvR-RBF (3 variables)----------------------
# column_names_list = ['Iteration', 'Log10(C)', 'Log10(Sigma)', 'AK']
# column_names_list = ['Iteration', 'log{0}(C)'.format('\u2081\u2080'), 'log{0}(σ)'.format('\u2081\u2080'), 'AK']
# svm_parallel_coor_3var_1(txt_file_path = '../../ml_sl/svm/ovr_txt_res/trained_on_tr_tested_on_vali/avg_res_and_plots/2020_06_25_svm_ovr_rbf_gs_avg_res.txt',
#                          column_names_list = column_names_list)
# ---------------------- Draw for SVM-OvR-RBF (3 variables)----------------------

# ---------------------- Draw for SVM-OvO-Poly(2) (3 variables)----------------------
# column_names_list = ['Iteration', 'Log10(C)', 'Log10(Quadratic Coefficient)', 'AK']
# 设置字符串的上下标显示
# Printing subscript and superscript in Python: https://codeigo.com/python/printing-subscript-and-superscript
# column_names_list = ['Iteration', 'log{0}(C)'.format('\u2081\u2080'),
#                      'log{0}(r)'.format('\u2081\u2080'), 'AK'] # r : Quadratic Coefficient
# svm_parallel_coor_3var_1(txt_file_path='../../ml_sl/svm/ovo_txt_res/trained_on_tr_tested_on_vali/avg_res_and_plots/2020_06_25_svm_ovo_poly_gs_avg_res.txt',
#                          column_names_list=column_names_list)
# ---------------------- Draw for SVM-OvO-Poly(2) (3 variables)----------------------

# ---------------------- Draw for SVM-OvR-Poly(2) (3 variables)----------------------
# column_names_list = ['Iteration', 'Log10(C)', 'Log10(Quadratic Coefficient)', 'AK']
column_names_list = ['Iteration', 'log{0}(C)'.format('\u2081\u2080'),
                     'log{0}(r)'.format('\u2081\u2080'), 'AK'] # r : Quadratic Coefficient
svm_parallel_coor_3var_1(txt_file_path='../../ml_sl/svm/ovr_txt_res/trained_on_tr_tested_on_vali/avg_res_and_plots/2020_06_25_svm_ovr_poly_gs_avg_res.txt',
                         column_names_list=column_names_list)
# ---------------------- Draw for SVM-OvR-Poly(2) (3 variables)----------------------

# ------------------------------------------- Draw parallel coordinates in version 1 -------------------------------------------