from GA_pack.genetic_material import Gene

class DNA():
    def __init__(self, gene_num_list, gene_range_pair_tuple_list, precision_list):
        """
        :param
            gene_list:
                [阻抗1， 阻抗2， 阻抗3，...] 有正负的浮点数
            gene_range_pair_tuple_list:
                [(阻抗1min, 阻抗1max), (阻抗2min, 阻抗2max), (阻抗3min, 阻抗3max), ...]
            precision_list:
                [阻抗1的要求精确到小数点后几位，阻抗2的要求精确到小数点后几位，阻抗3的要求精确到小数点后几位，...]
        """
        self.gene_num_list = gene_num_list
        self.gene_range_pair_tuple_list = gene_range_pair_tuple_list
        self.precision_list = precision_list

    def join_binary_code(self):
        gene_list = []
        for gene_num, gene_range_pair_tuple, precision in zip(self.gene_num_list, self.gene_range_pair_tuple_list, self.precision_list):
            gene_list.append(Gene.gene(num=gene_num,range_pair=gene_range_pair_tuple, precision=precision, encode_type='b').get_gene())
        return gene_list