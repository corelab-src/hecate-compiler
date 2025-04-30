import re
import pandas as pd
import csv
import sys
# import matplotlib.pyplot as plt
import math

#index : type
res_lists = {
        'benchname' : str,
        # 'library' : str,
        # 'device' : str,
        'compiler' : str,
        'epoch' : int,
        # 'sim': float,
        # 'waterline' : int,
        # 'packing' : int,
        # 'latency' : float,
        # 'rms' : float,
        'Execution Time' : float,
        'cst_size' : int,
        'hevm_size' : int,
        'total_file_size' : int,
        # 'encodeOnline' :int
        }

df = pd.DataFrame()
ress = []

conference = "ASPLOS25"
file_name = conference + '/codesize.txt'
csv_name = conference +'/csv/'+'codesize'+'.csv'

### for average data
option_list = ['benchname','compiler', 'epoch']
# result_mean = ['latency', 'Execution Time', 'rms', 'boot_time']
result_mean = []

f = open(file_name, 'r')
data=f.read()

data_dict = {}
print("Each Data Lenght should be same")
for name, ttype in res_lists.items() :
    if name == 'Execution Time':
        data_dict[name]=re.findall(r"%s: (.*) seconds" % name, data)
        print(name, len(data_dict[name]))
        continue
    data_dict[name]=re.findall(r"%s: (.*)" % name, data)
    print(name, len(data_dict[name]))
aa = list(zip(*list(data_dict.values())))

for i in range(len(aa)):
    target = aa[i]
    cnt = 0
    listt = []
    for name, ttype in res_lists.items() :
        data_result = ttype(target[cnt])
        listt.append(data_result)
        cnt += 1
        
    ress += [tuple(listt)]
print(ress)
f.close()

with open(csv_name, 'w',newline='') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    # write.writerow(res_lists)
    write.writerow(list(res_lists.keys()))
    write.writerows(ress)


# make average data
df = pd.read_csv(csv_name)
column_names = df.columns.tolist()
# df['boot_time'] = (df['boot_time'] / 1000000).round(6)
# print(df['boot_time'])

remaining_columns = [col for col in column_names if col not in result_mean]
result_mean_avg = [item + '_avg' for item in result_mean]
average_df = df.groupby(option_list)[result_mean].mean().reset_index()
df_unique = df.drop_duplicates(subset=option_list)
merged_df = pd.merge(df_unique, average_df, on=option_list, how='left', suffixes=('', '_avg'))
final_df = merged_df[remaining_columns+result_mean_avg]
final_df.columns = remaining_columns + result_mean

final_df.to_csv(csv_name, index=False)
print(final_df)
