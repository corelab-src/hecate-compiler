import re
import pandas as pd
import csv
import sys
# import matplotlib.pyplot as plt
import math

#index : type
res_lists = {
        'benchname' : str,
        'library' : str,
        'device' : str,
        'compiler' : str,
        # 'sim': float,
        'waterline' : int,
        'latency' : float,
        'rms' : float,
        'Execution Time' : float,
        # 'MemUsage': float,
        'cst_size' : int,
        'hevm_size' : int,
        'total_file_size' : int,
        'boot_cnt' : int,
        'epoch' : int,
        # 'encodeOnline' :int
        }

df = pd.DataFrame()
ress = []

conference = "ASPLOS25"
file_name = conference + '/total.txt'
csv_name = conference +'/csv/'+conference+'.csv'

### for average data
option_list = ['benchname','compiler', 'epoch' ]
result_mean = ['latency', 'Execution Time']

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
        listt.append(ttype(target[cnt]))
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


remaining_columns = [col for col in column_names if col not in result_mean]
result_mean_avg = [item + '_avg' for item in result_mean]
average_df = df.groupby(option_list)[result_mean].mean().reset_index()
df_unique = df.drop_duplicates(subset=option_list)
merged_df = pd.merge(df_unique, average_df, on=option_list, how='left', suffixes=('', '_avg'))
final_df = merged_df[remaining_columns+result_mean_avg]
final_df.columns = remaining_columns + result_mean

final_df.to_csv(csv_name, index=False)
