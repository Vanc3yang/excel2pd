import pandas as pd

import excel2pd

df = pd.DataFrame({'经度': [11, -2, 33, -4, 5],
                   '纬度': [-12, -2, 14, -5, 6],
                   '距离': [5, 2, 3, 9, 20],
                   '地址': ['K', 'K', '', '', '']})

configs = {'col1': 'ABS( MAX([经度]::) )',
       'col2': '[经度] / SUM( [纬度]::)',
       'col3': '[距离] - [经度]',
       'col4': 'SUM( [经度], [纬度])',
       'col5': 'MIN( [经度], [距离], [纬度]) + [距离]',
       'col6': 'ABS([距离] + [纬度]) + SUM([经度]::)',
       'col7': 'AVERAGE([经度]::)',
       'col8': 'COUNT([距离]::)',
       'col9': 'MEDIAN([经度]::)',
       'col10': 'MIN(MAX([经度]::), [纬度], MIN([距离]::))',
       'col11': 'IF([经度] > 5, \"yes\", \"no\")',
       'col12': 'LEN([经度])*5',
       'col13': 'IF([经度] == 5, 5, 6)',
       'col14': 'IF([地址] == \"\", \"yes\", \"no\")',
       'col15': 'STDEV.P([经度])',
       'col16': 'STDEV.S([经度])',

       }
df_new = excel2pd.Excel2Pd(configs, df).calc()
print(df_new)
