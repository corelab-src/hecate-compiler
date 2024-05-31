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

# file_name="./results/" + timestamp + ".txt"
# file_name = "dacapo_results"
conference = "ASPLOS25"
file_name = conference + '/total.txt'
csv_name = conference +'/csv/'+conference+'.csv'
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
# benchname = re.findall(r"benchname: (.*)", data)
# opt = re.findall(r"opt: (.*)", data)
# print(type(benchname))
# print(type(data_dict.values()))
# opt = re.findall(r"opt: (.*)", data)
# waterline = re.findall(r"waterline: (.*)", data)
# latency = re.findall(r"latency: (.*)", data)
# rms = re.findall(r"rms: (.*)", data)
# aa = list(zip(benchname, opt, comp_time, waterline, latency,  rms, est_lat, est_error, init_level))
# aa = list(zip(benchname, opt, waterline, total, bootstrap, bootcnt, esticost, compilet, boot_mgmt, opnum, numcand, threshold, rms))
aa = list(zip(*list(data_dict.values())))
# print(aa)
iternum = 1
removenum = 0;

# print(aa)


for i in range(0, len(aa), iternum):
    target = aa[i]
    # tot_list = []
    # boot_list = []
    # rmss_list = []
    # for k in range(0, iternum) :
    #     tot_list.append(float(aa[i+k][3]))
    #     # boot_list.append(int(aa[i+k][4]))
    #     # rmss_list.append(float(aa[i+k][5]))
    # for k in range(0, removenum) :
    #     id = tot_list.index(max(tot_list))
    #     tot_list.remove(tot_list[id])
    #     # boot_list.remove(boot_list[id])
    #     rmss_list.remove(rmss_list[id])
    # tot = sum(tot_list) / (iternum - removenum)
    # boot = sum(boot_list) / (iternum - removenum)
    # rmss = sum(rmss_list) / (iternum - removenum)

    # tot = int(target[5])
    # boot = int(target[6])
    # rmss = math.log10(float(target[6]))
    # rmss = math.log10(rmss)


    cnt = 0
    listt = []
    for name, ttype in res_lists.items() :
        listt.append(ttype(target[cnt]))
        cnt += 1
        
    ress += [tuple(listt)]
    # ress += [(name,  opt,comptime, waterline, tot, math.log(rmss, 2), est_lat, math.log(est_error, 2), init_level)]
print(ress)
f.close()
with open(csv_name, 'w',newline='') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    # write.writerow(res_lists)
    write.writerow(list(res_lists.keys()))
    write.writerows(ress)



