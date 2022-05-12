# SEIQRmap
 SEQIRmap为论文的代码包，提供了获取数据，模型计算，可视化等功能。所有的计算方法和模型均基于所发表的论文。test1-test2

## 1. SEIQR.get_data(country,t_start_str,t_end_str) 
该函数可获取计算所需要的数据包括每日新增感染人数，每日新增感染人数，累计感染人数，每日疫苗接种率等，数据预处理的方法均采用论文中的方法。 
| 参数 | 说明 | 数据类型 |
| :----: | :----: | :----: |
| country | 国家的英文名称(首字母应该大写) | str |
| t_start_str | 开始日期,格式为'xxxx-xx-xx'如'2022-01-01' | str | 
| t_end_str | 结束日期,格式为'xxxx-xx-xx'如'2022-01-01' | str |

| 返回值 | 说明 | 数据类型 |
| :----: | :----: | :----: |
| data | 获取并经过预处理的数据 | dataframe |
| population | 该国总人口 | int |

## 2. SEIQR.fit_predict(country,t_start_str,t_end_str,days)
拟合并预测某一国家的感染人数。此函数基于最小二乘最优选择参数，初值为开始日期的数据，无需自定义参数。由于潜伏者E需要根据4天后的易感者I的人数估算，所以拟合最多支持到当天的前四天。预测为根据最小二乘法计算出的最优参数进行计算，若预测时间较长可能会不准确,建议最长不超过3个月。
| 参数 | 说明 | 数据类型 |
| :----: | :----: | :----: |
| country | 国家的英文名称(首字母应该大写) | str |
| t_start_str | 拟合开始日期,格式为'xxxx-xx-xx'如'2022-01-01' | str | 
| t_end_str | 拟合结束日期,格式为'xxxx-xx-xx'如'2022-01-01' | str |
| days | 向后预测的天数 | int |

| 返回值 | 说明 | 数据类型 |
| :----: | :----: | :----: |
| result_I | 拟合和预测的感染人数的结果 | dataframe |
| r2 | r^2用来评估模型拟合的效果 | int |
| result_para | 最小二乘法计算出的最优参数 | dataframe |

## 3. SEIQR.fit_predict_multi(country,t_start_str_list,t_end_str_list,days)
该函数适用于对某一国家进行长期拟合的情况，可将某一国家分为多段进行预测。是对对该模型长期拟合能力不足的改进，通过合理的分段可以得到很好的拟合效果。
| 参数 | 说明 | 数据类型 |
| :----: | :----: | :----: |
| country | 国家的英文名称(首字母应该大写) | str |
| t_start_str_list | 拟合开始日期列表，将分段的所有开始日期放在此列表中 | list | 
| t_end_str_list | 拟合结束日期列表，将分段的所有结束日期放在此列表中 | list |
| days | 向后预测的天数 | int |

| 返回值 | 说明 | 数据类型 |
| :----: | :----: | :----: |
| result_I | 拟合和预测的感染人数的结果 | dataframe |
| r2_list | 各阶段的r^2 | list |
| result_para | 各阶段的最优参数 | dataframe |

## 4. SEIQR.SEIQR(country_list,t_start_str,t_end_str,days)
可视化函数，若不想知道计算的细节，可直接使用此函数。该函数可将指定的一些国家在某段时间内的感染人数变化情况映射到地图上，地图基于pyecharts开发，提供可交互的地图和动画化展示。运行完成后将会在程序运行文件目录下生成SEIQR_map.html文件。
| 参数 | 说明 | 数据类型 |
| :----: | :----: | :----: |
| country_list | 国家的英文名称(首字母应该大写)组成的列表 | list |
| t_start_str | 拟合开始日期,格式为'xxxx-xx-xx'如'2022-01-01' | str | 
| t_end_str | 拟合结束日期,格式为'xxxx-xx-xx'如'2022-01-01' | str |
| days | 向后预测的天数 | int |
