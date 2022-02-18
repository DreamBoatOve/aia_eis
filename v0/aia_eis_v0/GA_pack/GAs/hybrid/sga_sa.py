from GA_pack.GAs.basic.sga import SGA
from GA_pack.encoders.binary_en_de_coder import multi_binary_encoder, multi_binary_list_decoder
from SA_pack.energy_functions.energy_pack import energy_pack
from SA_pack.SAs.sa import sa_thread
import time
from queue import Queue
from threading import Thread

class SGA_SA(SGA):
    def __init__(self, paras_path, fitness_type, dna_num, max_epoch, p_crossover, p_mutation):
        SGA.__init__(self, paras_path, fitness_type, dna_num, max_epoch, p_crossover, p_mutation)
    def evolution(self, relate_res_folder_path):
        terminated_condition = True
        DNAs_binary_list = self.DNAs_random_initialize()
        DNAs_fitness_list = self.get_fittness(DNAs_binary_list)
        epoch = 0
        # 保存每次进化得到的个体的信息（进化代数，最优个体（变量值，适应度），变量的值-二进制转浮点数，适应度）
        # 获取当前时间的字符串--'19_01_27_22_30'
        time_str = time.strftime('%y_%m_%d_%H_%M', time.localtime(time.time()))
        db_name = relate_res_folder_path + '/sga_sa' + time_str
        while terminated_condition:
            self.serialize(db_name, epoch, DNAs_binary_list, DNAs_fitness_list)
            selected_DNAs_list = self.select(DNAs_binary_list, DNAs_fitness_list, select_type='proportion')
            crossed_DNAs_list = self.crossover(selected_DNAs_list, crossover_type='single')
            DNAs_binary_list = self.mutation(crossed_DNAs_list, mutation_type='simple')
            DNAs_fitness_list = self.get_fittness(DNAs_binary_list)
            """
            遗传+模拟退火
                把二进制的DNA列表转化为十进制的DNA列表，
                每个DNA0作为一个模拟退火的初值，并得到一个退火后的DNA1
                比较DNA0和DNA1的适应度，取适应度高的DNA进入下一代进化
            """
            DNAs_decimal_list = multi_binary_list_decoder(DNAs_binary_list, self.gene_range_pair_tuple_list, self.precision_list).decode()
            thread_index = 0
            res_queue_len = len(DNAs_decimal_list)
            res_queue = Queue(maxsize=res_queue_len)
            for DNA_decimal_list in DNAs_decimal_list:
                t = Thread(target=sa_thread, args=(thread_index, res_queue, DNA_decimal_list, energy_pack, self.fitness_type, self.gene_range_pair_tuple_list, self.precision_list))
                t.start()
                print('Thread-{} starts running'.format(thread_index))
                thread_index += 1
            while res_queue.qsize() < res_queue_len:
                # time.sleep(2)
                print('Still have {} threads unfinished'.format(res_queue_len - res_queue.qsize()))
            while res_queue.qsize() > 0:
                thread_index, optimal_state_list, optimal_state_energy = res_queue.get()
                if DNAs_fitness_list[thread_index] < -optimal_state_energy:
                    DNAs_binary_list[thread_index] = multi_binary_encoder(optimal_state_list, self.gene_range_pair_tuple_list, self.precision_list).encode()
            epoch += 1
            terminated_condition = self.termination_judge(DNAs_fitness_list, epoch, min_MSE_threshold=1)
if __name__ == '__main__':
    sga_sa = SGA_SA(paras_path='../../../parameters/rosenbrock_paras.txt', fitness_type='rosenbrock', dna_num=80, max_epoch=200, p_crossover=0.6, p_mutation=0.001)
    sga_sa.evolution(relate_res_folder_path='../../../results/sga_sa_results')