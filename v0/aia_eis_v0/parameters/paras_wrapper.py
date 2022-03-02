import shelve
import ast

class para_serializer:
    """
    存取变量的相关参数,变量的变化范围,精度
    在读取文件名如r(qr)时，文件中内容存放的顺序与文件名的顺序对应
    """
    def __init__(self, para_file_name):
        self.para_file_name = para_file_name

    def read_raw_para(self):
        gene_range_pairs_tuple_list = []
        precisions_list = []
        with open(self.para_file_name, 'r') as para_file:
            for line in para_file.readlines():
                gene_range_pair_list_str, precision_str = line.strip().split(' ')
                # python将字符串类型list转换成list--https://www.cnblogs.com/TTyb/p/9717546.html
                gene_range_pair_list = ast.literal_eval(gene_range_pair_list_str)
                precision_list = ast.literal_eval(precision_str)
                gene_range_pairs_tuple_list.append(gene_range_pair_list)
                precisions_list.append(precision_list)
        return gene_range_pairs_tuple_list, precisions_list

# if __name__ == '__main__':
#     gene_range_pairs_tuple_list, precisions_list = para_serializer('rosenbrock_paras.txt').read_raw_para()
#     print(gene_range_pairs_tuple_list, precisions_list)