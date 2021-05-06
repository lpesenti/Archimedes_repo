import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import os

path_to_data = r"D:\Archimedes\Data"
all_data = []
final_df = pd.DataFrame()
data_folder = r"D:\Archimedes\Data\SosEnattos_Data_20210219"
temp_data = glob.glob(os.path.join(data_folder, "*.lvm"))
temp_data.sort(key=lambda f: int(re.sub('\D', '', f)))
all_data += temp_data
i = 0
for data in all_data:
    print(round(i / len(all_data) * 100, 1), '%')
    # Read only the column of interest -> [index]
    a = pd.read_table(data, sep='\t', low_memory=False, header=None)
    # At the end we have a long column with all data
    final_df = pd.concat([final_df, a], axis=0, ignore_index=True)
    i += 1
header_lst = []
for i in final_df.columns:
    header_lst.append(str(i))
print('New header created')
final_df.columns = header_lst
print('New header inserted')
# final_df.to_parquet(r"D:\Archimedes\Data\20210219.parquet.gzip", compression='gzip', compression_level=9)
# final_df.to_parquet(r"D:\Archimedes\Data\20210219.parquet.snappy", compression='snappy')
# final_df.to_parquet(r"D:\Archimedes\Data\20210219.parquet.brotli", compression='brotli', compression_level=9)
print('File saved')
df_test = pd.read_parquet(r"D:\Archimedes\Data\20210219.parquet.brotli")
print(df_test.equals(final_df))
x = np.arange(len(df_test))
y1 = df_test['1'] - final_df['1']
y2 = df_test['2'] - final_df['2']

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax1.plot(x, y1)

fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.plot(x, y2)

plt.show()
