class binary_encoder():
    """
    根据对一个变量的范围、精度、二进制编码类型进行编码，得到一个只包含0，1的list
    """
    def __init__(self, num, range_pair, precision):
        """
        :param num: 用浮点数表示的变量的原本数值
        :param range_pair: tuple(lower_bound, upper_bound)
        :param precision: int, 数据要求精确到小数点后几位
        :param encoding_length: 变量用多长的二进制表示
        :param step_length: 区间被二进制划分后，每一小段的长度
        """
        self.num = num
        self.range_pair = range_pair
        self.precision = precision
        self.encoding_length, self.step_length = self.get_encoding_length()
    def get_encoding_length(self):
        range = self.range_pair[1] - self.range_pair[0]
        encoding_length = 1
        condition = True
        while condition:
            if range * (10 ** self.precision) <= (2**encoding_length-1):
                condition = False
            else:
                encoding_length += 1
        step_length = range/(2**encoding_length-1)
        return encoding_length, step_length
    def encode(self):
        binary_str = format(int((self.num - self.range_pair[0])/self.step_length), 'b')
        binary_list = [0 for i in range(self.encoding_length - len(binary_str))] + list(map(int, binary_str))
        return binary_list
class multi_binary_encoder:
    def __init__(self, nums_list, range_pairs_list, precisions_list):
        self.nums_list = nums_list
        self.range_pairs_list = range_pairs_list
        self.precisions_list = precisions_list
    def encode(self):
        multi_binary_list = []
        for num, range_pair, precision in zip(self.nums_list, self.range_pairs_list, self.precisions_list):
            multi_binary_list.append(binary_encoder(num, range_pair, precision).encode())
        return multi_binary_list
class binary_decoder():
    """
    根据二进制的编码、范围、精度，转化为对应的浮点数
    """
    def __init__(self, binary_list, range_pair, precision):
        """
        :param binary_list: list 只包含0，1的二进制编码，例如：[0,1,0,1,0]
        :param range_pair: tuple(lower_bound, upper_bound)
        :param precision: int, 数据要求精确到小数点后几位
        """
        self.binary_list = binary_list
        self.range_pair = range_pair
        self.precision = precision
        self.encoding_length, self.step_length = self.get_encoding_length()
    def get_encoding_length(self):
        range = self.range_pair[1] - self.range_pair[0]
        encoding_length = 1
        condition = True
        while condition:
            if range * (10 ** self.precision) <= (2 ** encoding_length - 1):
                condition = False
            else:
                encoding_length += 1
        step_length = range / (2 ** encoding_length - 1)
        return encoding_length, step_length
    def decode(self):
        steps = 0
        self.binary_list.reverse()
        for index, b in enumerate(self.binary_list):
            steps += b * (2**index)
        num = self.range_pair[0] + steps * self.step_length
        return num
class multi_binary_list_decoder:
    def __init__(self, multi_binary_list, range_pairs_list, precisions_list):
        self.multi_binary_list = multi_binary_list
        self.range_pairs_list = range_pairs_list
        self.precisions_list = precisions_list
    def decode(self):
        nums_list = []
        for binary_list in self.multi_binary_list:
            num_list = []
            for gene_list, range_pair, precision in zip(binary_list, self.range_pairs_list, self.precisions_list):
                num_list.append(binary_decoder(gene_list, range_pair, precision).decode())
            nums_list.append(num_list)
        return nums_list
# if __name__ == '__main__':
#     be = binary_encoder(num=12,range_pair=(0,15),precision=0)
#     binary_list = be.encode()
#     print('Num to binary:\t',binary_list)
#     bd = binary_decoder(binary_list, range_pair=(0,15), precision=0)
#     num = bd.decode()
#     print('Binary to num:\t',num)