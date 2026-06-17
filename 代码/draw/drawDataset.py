import ujson
import os
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['font.size'] = 25  # 设置所有字体的默认大小
# 文件路径和保存路径
base_path = "G:\联邦\新建文件夹\图片\数据集"
config_files = ["Fashion-MNIST-config1.json","CIFAR-10-config1.json"]
savepath = os.path.join(base_path, "final322-song1.pdf")

# 颜色设置
colors = ['#dab49d', '#adc178', '#93b7be', '#457b9d', '#0a9396', '#ee9b00', '#e9c46a',
          '#fee440', '#f7e1d7', '#f07167', '#6d6875', '#bcb8b1', '#a49fd5', '#7FFF00']

# 读取数据并计算统一的最大值
max_height = 0
data_list = []
for file_name in config_files:
    file_path = os.path.join(base_path, file_name)
    with open(file_path, "r") as f:
        config = ujson.load(f)
        data = config["statistic"]
        data_list.append(data)
        for user_data in data:
            total_height = sum(int(user_data[str(tag)]) if str(tag) in user_data else 0 for tag in range(13))
            max_height = max(max_height, total_height)

# 创建 2x2 子图
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes = axes.flatten()

# 遍历数据和子图
for idx, (data, ax) in enumerate(zip(data_list, axes)):
    # 数据
    labels = data
    tags = list(range(0, 10))

    # 柱在图横轴的位置
    x = np.arange(len(labels))
    cumulative_heights = np.zeros(len(labels))

    # 绘制堆叠柱状图
    for i, tag in enumerate(tags):
        sample_counts = [user_data[str(tag)] if str(tag) in user_data else 0 for user_data in data]
        ax.bar(x, sample_counts, width=0.5, label=f"类别 {tag}", color=colors[i], bottom=cumulative_heights)
        cumulative_heights += np.array(sample_counts)

    ax.set_xlabel("客户端序号")  # 调整字体大小
    ax.set_ylabel("样本量")  # 调整字体大小
    ax.set_xticks(x)
    ax.set_xticklabels(np.arange(0, len(labels)))  # 设置x轴字体大小
    ax.set_ylim(0, max_height * 1.1)  # 统一纵坐标范围

    # 设置子图标题放在下方
    ax.text(0.5, -0.16, f"({chr(97 + idx)}) {config_files[idx].replace('-config1.json', ' ')}",
            ha='center', va='center', transform=ax.transAxes,fontname='Arial')

# 添加全局标题
# fig.suptitle('不同配置下的KronoDroid数据集样本分布', fontsize=20)

handles, labels = axes[0].get_legend_handles_labels()
# 调整图例位置参数
fig.legend(handles, labels,
           loc='lower center',
           bbox_to_anchor=(0.5, 0.0),  # 从0.02提升到0.06
           ncol=5,
           frameon=False,
           borderaxespad=0.5)  # 增加图例与标题间距

# 重新分配画布空间
plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # 底部空间从8%增加到12%
plt.savefig(savepath, format='pdf')
plt.show()