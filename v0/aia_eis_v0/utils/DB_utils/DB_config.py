import platform

"""
我有三个机器，每个机器的Mysql设置不完全一样，在此先查找当前是哪个机器，再返回相应的SQL配置信息
"""

def mysql_config():
    '''
    计算机的网络名称
        Desktop: 'ADMIN-20160708U'
        NH55: ''
        G505: ''
    '''
    pc_name = platform.node()
    if pc_name == 'ADMIN-20160708U':
        mysql_config_str = ''
        return mysql_config_str