"""
非时间一致性

"""



# 1、读取数据
import pandas as pd
import  torch
import numpy as np

root_Path ="F:\\兆南师兄\\数据集\\kronodroid-main\\kronodroid-main\\real_device\\real_malware_v1\\"
fileName ="real_malware_v1"
dataPath =root_Path + fileName+".csv"

print("处理",dataPath,"中...")
df_Mal = pd.read_csv(dataPath)
print("原始恶意软件数据大小",df_Mal.shape)
# 2. 删除空值
df_Mal = df_Mal.dropna()
print("删除空值之后",df_Mal.shape)

# 3. 删除不合理的数据
# 3-1 删除不正确的的时间格式（1980）
df_Mal = df_Mal[~df_Mal.isin(["1980-00-00"]).any(axis=1)]
print("删除1980之后",df_Mal.shape)

# 3-2 把时间小于08年的都删除
df_Mal['HighestModDate'] = pd.to_datetime(df_Mal['HighestModDate'], format='%m/%d/%Y')
start_date = pd.to_datetime('01/01/2008', format='%m/%d/%Y')
df_Mal = df_Mal[df_Mal['HighestModDate'] > start_date]
print("截断到08年之后",df_Mal.shape)

# 3-3 把检测率=0、家族名为空的删除（上面那些步骤已经相当于做了这个了）
# 3-4 把属于两个家族的都删除
# 后续可以多标签分类、或者是引入不确定性
df_Mal = df_Mal[~df_Mal['MalFamily'].str.contains('/')]
print("把属于两个家族的都删除之后",df_Mal.shape)








# 接下来处理良性样本

# 1、读取数据

root_Path ="F:\\兆南师兄\\数据集\\kronodroid-main\\kronodroid-main\\real_device\\real_legitimate_v1\\"
fileName ="real_legitimate_v1"
dataPath =root_Path + fileName+".csv"

print("处理",dataPath,"中...")
df_Benign = pd.read_csv(dataPath)
print("原始良性软件数据大小",df_Benign.shape)


# 2. 删除MalFamliy非空的行
df_Benign = df_Benign[df_Benign['Detection_Ratio']==0]
df_Benign = df_Benign[df_Benign['MalFamily'].isna()]
print("删除MalFamliy有值的之后",df_Benign.shape)

# 3. 删除不合理的数据
# 3-1 删除不正确的的时间格式（1980）
df_Benign = df_Benign[~df_Benign.isin(["1980-00-00"]).any(axis=1)]
print("删除1980之后",df_Benign.shape)

# 3-2 把时间小于08年的都删除
df_Benign['HighestModDate'] = pd.to_datetime(df_Benign['HighestModDate'], format='%m/%d/%Y')
start_date = pd.to_datetime('01/01/2008', format='%m/%d/%Y')
df_Benign = df_Benign[df_Benign['HighestModDate'] > start_date]
df_Benign['MalFamily']='benign'
df_Benign = df_Benign.dropna()
print("截断到08年之后",df_Benign.shape)





# 合并良性+恶意
# df_Mal.reset_index(drop=True, inplace=True)
# df_Benign.reset_index(drop=True, inplace=True)

merged_df = pd.concat([df_Mal, df_Benign], axis=0)
print("合并之后的数据集：",merged_df.shape)
merged_df_temp = merged_df.copy()
dropList =['Malware','Package','sha256','EarliestModDate','HighestModDate','Detection_Ratio','MalFamily']
for listName in dropList:
    merged_df_temp.drop(listName, axis=1, inplace=True)
print("删除不用的列之后的临时数据集：",merged_df_temp)

variance = merged_df_temp.var()
merged_df_temp = merged_df_temp.loc[:, variance != 0]
print("删除方差=0的特征之后的临时数据集",merged_df_temp.shape)


# 3-6 标准化
# mean = merged_df_temp.mean()
# std = merged_df_temp.std()
# merged_df_temp = (merged_df_temp - mean) / std



# 把sha256 和时间戳 还有家族加上去
selected_columns = merged_df[['Malware','sha256', 'HighestModDate','MalFamily']]
merged_df_temp = pd.concat([merged_df_temp, selected_columns], axis=1)
merged_df = merged_df_temp
print("经过特征筛选的数据集(恶意+良性）：",merged_df.shape)


# 拆成
df_Benign = merged_df[merged_df['Malware']==0]
df_Benign['MalFamily']='benign'
df_Mal = merged_df[merged_df['Malware']==1]

print("经过特征筛选的数据集(良性）：",df_Benign.shape)
print("经过特征筛选的数据集(恶意）：",df_Mal.shape)

# # 4. 按照时间戳拆分成两个dataframe
# split_date = pd.to_datetime('07/01/2017', format='%m/%d/%Y')
# df_Mal_train = df_Mal[df_Mal['HighestModDate'] < split_date]
# df_Mal_test = df_Mal[df_Mal['HighestModDate'] >= split_date]
# print('以',split_date,'为分界线，恶意软件训练集的大小：',df_Mal_train.shape)
# print('以',split_date,'为分界线，恶意软件测试集的大小：',df_Mal_test.shape)


part_one_fraction = 2 / (2 + 1)  # 这是2/3，因为2:1总和为3
part_one_num = int(len(df_Mal) * part_one_fraction)
df_Mal_train = df_Mal.sample(n=part_one_num)
df_Mal_test = df_Mal.drop(df_Mal_train.index)
print('以',part_one_fraction,'为分界线，恶意软件训练集的大小：',df_Mal_train.shape)
print('以',part_one_fraction,'为分界线，恶意软件测试集的大小：',df_Mal_test.shape)


df_Benign_train = df_Benign.sample(n=part_one_num)
df_Benign_test = df_Benign.drop(df_Benign_train.index)
print('以',part_one_fraction,'为分界线，良性软件训练集的大小：',df_Benign_train.shape)
print('以',part_one_fraction,'为分界线，良性软件测试集的大小：',df_Benign_test.shape)

# df_Benign_train = df_Benign[df_Benign['HighestModDate'] < split_date]
# df_Benign_test = df_Benign[df_Benign['HighestModDate'] >= split_date]
# print('以',split_date,'为分界线，良性软件训练集的大小：',df_Benign_train.shape)
# print('以',split_date,'为分界线，良性软件测试集的大小：',df_Benign_test.shape)

# df_Benign_train = df_Benign_train.sample(n=2500)
# df_Benign_test = df_Benign_test.sample(n=1200)


# # 5. 统计两个dataframe中每个家族的数据量
# df_Mal_train_familyAndNum = df_Mal_train['MalFamily'].value_counts()
# # print("训练集家族统计：",df_Mal_train_familyAndNum)
# df_Mal_family_greater_than_50 = df_Mal_train_familyAndNum[df_Mal_train_familyAndNum > 55]
# print("筛选后恶意软件训练集家族统计：",df_Mal_family_greater_than_50)
# df_Mal_test_familyAndNum = df_Mal_test['MalFamily'].value_counts()
# # print("测试集家族统计：",df_Mal_test_familyAndNum)
# df_Mal_test_restricted = df_Mal_test_familyAndNum[df_Mal_test_familyAndNum.index.isin(df_Mal_family_greater_than_50.index)]
# print("筛选后恶意软件测试集家族统计：",df_Mal_test_restricted)
#
# df_Mal_train = df_Mal_train[df_Mal_train['MalFamily'].isin(df_Mal_family_greater_than_50.index)]
# df_Mal_test = df_Mal_test[df_Mal_test['MalFamily'].isin(df_Mal_test_restricted.index)]

df_Mal_train_familyAndNum = df_Mal_train['MalFamily'].value_counts()
# print("训练集家族统计：",df_Mal_train_familyAndNum)
df_Mal_family_greater_than_50 = df_Mal_train_familyAndNum[df_Mal_train_familyAndNum >= 500]
# print("筛选后恶意软件训练集家族统计：",df_Mal_family_greater_than_50)
df_Mal_test_familyAndNum = df_Mal_test['MalFamily'].value_counts()
common_indices = df_Mal_test_familyAndNum.index.intersection(df_Mal_family_greater_than_50.index)
# print(common_indices)
common_values_in_test = df_Mal_family_greater_than_50[common_indices]
common_values_in_train = df_Mal_train_familyAndNum[common_indices]
print(common_values_in_test)
print(common_values_in_train)
# print("测试集家族统计：",df_Mal_test_familyAndNum)
# df_Mal_train_restricted = df_Mal_train_familyAndNum[df_Mal_test_familyAndNum.index.isin(common_values_in_train.index)]
# df_Mal_test_restricted = df_Mal_test_familyAndNum[df_Mal_test_familyAndNum.index.isin(common_values_in_test.index)]


df_Mal_train = df_Mal_train[df_Mal_train['MalFamily'].isin(common_values_in_train.index)]
df_Mal_test = df_Mal_test[df_Mal_test['MalFamily'].isin(common_values_in_test.index)]


# sample_a = df_Mal_test[df_Mal_test['MalFamily'] == 'SMSreg'].sample(n=1000, random_state=1)  # 从Family为'A'的行中随机选择150个
# filtered_df = df_Mal_test[df_Mal_test['MalFamily'] != 'SMSreg']
# # 将各部分的结果合并成一个新的DataFrame
# df_Mal_test = pd.concat([sample_a, filtered_df])



print("筛选后恶意软件训练集家族统计：", df_Mal_train.shape)
print("筛选后恶意软件测试集家族统计：", df_Mal_test.shape)
#
#
# # (5. 删除冗余特征)
#
#
#
#
#
#
# # 6. 给MalFamily编号
familyName = df_Mal_train['MalFamily']
mapping = dict()
counter = 1 #恶意软件从1开始映射
for value in familyName:
    if value not in mapping:
        mapping[value] = counter
        counter += 1
df_Mal_train['MalFamilyID'] = df_Mal_train['MalFamily'].map(mapping)
df_Mal_test['MalFamilyID'] = df_Mal_test['MalFamily'].map(mapping)
print("恶意软件最终的训练集：",df_Mal_train.shape)
print("恶意软件最终的测试集：",df_Mal_test.shape)



df_Benign_train['MalFamilyID'] = 0
df_Benign_test['MalFamilyID'] = 0
print("良性软件最终的训练集：",df_Benign_train.shape)
print("良性软件最终的测试集：",df_Benign_test.shape)



# 合并恶意+良性作为训练集、测试集
train_All = pd.concat([df_Benign_train, df_Mal_train], axis=0)
test_All = pd.concat([df_Benign_test, df_Mal_test], axis=0)
print("最终的训练集：",train_All.shape)
print("最终的测试集：",test_All.shape)


#
#
#
def convertMatrix(feature):
    feature = list(feature)
    # print(feature)
    # total_sum = np.sum(feature)
    # print(total_sum)
    for p in range(34):
        feature.append(0)
    vector = np.array(feature)
    # vector = standardize(vector)
    # matrix = np.reshape(vector,(15,15)) #互信息法（没有删除方差）
    # matrix = np.reshape(vector,(14,14))  #互信息法（删除方差）
    matrix = np.reshape(vector,(18,18))  #互信息法（删除方差）
    return matrix

k=0
# 7. 转为图像
for df in [train_All,test_All]:
    features =df.iloc[:, :290].values
    hash = df['sha256'].values
    time = df['HighestModDate'].values
    malFamily = df['MalFamily'].values
    malFamilyID =df['MalFamilyID'].values
    malware=df['Malware'].values
    # 创建一个空字典来存储结果
    # train_dict = {'X': [],'Y': [], 'sha256': [],'time': [],'FamilyName': []}

    train_x,train_y,train_hash,train_time,train_FamilyName,train_malware=[],[],[],[],[],[]
    for i in range(len(df)):
        # print(i)
        # data = np.array(features[i].reshape(16, 16))
        data = convertMatrix(features[i])
        train_x.append(data)
        train_y.append(np.array(malFamilyID[i]))
        train_hash.append(np.array(hash[i]))
        train_time.append(np.array(time[i]))
        train_FamilyName.append(np.array(malFamily[i]))
        train_malware.append(np.array(malware[i]))

    train_x = np.stack(train_x)
    train_y = np.array(train_y)
    train_name = np.array(train_hash)
    train_time = np.array(train_time)
    train_FamilyName = np.array(train_FamilyName)
    train_malware= np.array(train_malware)
    if (k==0):
        train_dict = {
            'X': train_x,
            'Y': train_y,
            'sha256': train_name,
            'time': train_time,
            'familyName':train_FamilyName,
            'malware':train_malware
        }
    else:
        test_dict = {
            'X': train_x,
            'Y': train_y,
            'sha256': train_name,
            'time': train_time,
            'familyName':train_FamilyName,
            'malware': train_malware
        }
    k = k+1

# # 8. 保存为npy文件
# print("处理",dataPath,"完成...")
#

save_Path ="F:\\兆南师兄\\数据集\\kronodroid-main\\kronodroid-main\\real_device\\"
train_save_path = save_Path+'train_notime.npy'
test_save_path = save_Path+'test_notime.npy'
np.save(train_save_path, train_dict)
np.save(test_save_path, test_dict)
train_All_family = train_All['MalFamilyID'].value_counts()
test_All_family = test_All['MalFamilyID'].value_counts()
print(train_All_family)
print(test_All_family)
train_All.to_csv('F:\\兆南师兄\\数据集\\kronodroid-main\\kronodroid-main\\real_device\\train_All.csv', index=False)
test_All.to_csv('F:\\兆南师兄\\数据集\\kronodroid-main\\kronodroid-main\\real_device\\test_All.csv', index=False)












# 4. 按照时间戳拆分成两个dataframe


# # 3-5 把不用的信息（比如sha、不用的时间戳去除）
# df_Benign_temp = df_Benign.copy()
# dropList =['Package','sha256','EarliestModDate','HighestModDate','Detection_Ratio','MalFamily']
# for listName in dropList:
#     df_Benign_temp.drop(listName, axis=1, inplace=True)
# print("删除不用的列之后的临时数据集：",df_Benign_temp)
# variance = df_Benign_temp.var()
# df_Benign_temp = df_Benign_temp.loc[:, variance != 0]
# print("删除方差=0的特征之后的临时数据集",df_Benign_temp.shape)
#
#
#
# # 3-6 标准化
# mean = df_Benign_temp.mean()
# std = df_Benign_temp.std()
# df_Benign_temp = (df_Benign_temp - mean) / std
#
#
#
# # 把sha256 和时间戳 还有家族加上去
# selected_columns = df_Benign[['sha256', 'HighestModDate','MalFamily']]
# df_Benign_temp = pd.concat([df_Benign_temp, selected_columns], axis=1)
# df_Benign_temp['MalFamily'] = 'benign'
# df_Benign = df_Benign_temp
# print("经过特征筛选的数据集：",df_Benign.shape)

