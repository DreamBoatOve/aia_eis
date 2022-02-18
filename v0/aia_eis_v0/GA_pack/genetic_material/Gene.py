from GA_pack.encoders import binary_en_de_coder

class gene():
    """
    根据对一个变量的范围、精度、编码类型进行编码
    """
    def __init__(self, num, range_pair, precision, encode_type):
        """
        :param num: 用浮点数表示的变量的原本数值
        :param range_pair: tuple(lower_bound, upper_bound)
        :param precision: int, 数据要求精确到小数点后几位
        :param encode_type: string, 'b': binary; 'f': float;
        """
        self.num = num
        self.range_pair = range_pair
        self.precision = precision
        self.encode_type = encode_type
    def get_gene(self):
        if self.encode_type == 'b':
            binary_list = binary_en_de_coder.binary_encoder(num=self.num, range_pair=self.range_pair, precision=self.precision).encode()
        elif self.encode_type == 'f':
            # 浮点数编码暂时不考虑，先空着
            pass
        encoded_list = binary_list
        return encoded_list