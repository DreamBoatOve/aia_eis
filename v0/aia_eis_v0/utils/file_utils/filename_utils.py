import datetime

def get_date_prefix():
    year_str = str(datetime.datetime.now().year)
    month = datetime.datetime.now().month
    day = datetime.datetime.now().day

    if month < 10:
        month_str = '0' + str(month)
    else:
        month_str = str(month)

    if day < 10:
        day_str = '0'+str(day)
    else:
        day_str = str(day)
    # date_str, like '2020_04_01_'
    date_str = year_str + '_'+month_str+'_'+day_str+'_'
    return date_str

# if __name__ == '__main__':
#     date_str = get_date_prefix()
#     print(date_str) # 2020_04_12_

def get_ecm_num_str(ecm_num):
    ecm_num_str = ''
    if ecm_num < 10:
        ecm_num_str += '00' + str(ecm_num)
    elif ecm_num < 100:
        ecm_num_str += '0' + str(ecm_num)
    else:
        return str(ecm_num)
    return ecm_num_str

def get_Num_len(num, length):
    num_str = ''
    if length == 2:
        if num < 10:
            num_str += '0' + str(num)
        elif num < 100:
            num_str += str(num)
    elif length == 3:
        if num < 10:
            num_str += '00' + str(num)
        elif num < 100:
            num_str += '0' + str(num)
    return num_str