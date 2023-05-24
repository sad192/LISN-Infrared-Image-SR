# 测试
# 开发时间：2023/3/2 20:40
import matplotlib.pyplot as plt
import numpy as np
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rc('font',family='Times New Roman')
y_axis_data_x4 = [38.97, 39.21, 39.43, 39.44, 39.51, 39.67, 39.66,39.67]  # y
## [43.01, 43.10, 43.12, 43.31, 43.46, 43.64, 43.69,43.68]
x_axis_data = [800, 1600, 2400, 3200, 4000, 4800, 5600,6000]  # x
y_axis_data_x2 = [43.01, 43.10, 43.12, 43.31, 43.46, 43.64, 43.69,43.68]  # y
swin_x2 = [43.07,43.18,43.26,43.24,43.34,43.49,43.57,43.55]
swin_x4 = [39.06,39.20,39.42,39.46,39.53,39.51,39.48,39.52]

imdn_x2 = [43.11,43.19,43.33,43.42,43.32,43.43,43.41,43.40]
imdn_x4 = [39.01,39.40,39.43,39.48,39.47,39.52,39.51,39.52]

dpsr_x2 = [42.23,42.69,42.99,42.74,42.97,43.14,43.09,43.11]
dpsr_x4 = [38.82,38.73,39.33,39.33,39.35,39.37,39.36,39.37]

pan_x2 = [43.01,43.06,43.07,43.07,43.08,43.08,43.10,43.09]
pan_x4 = [39.29,39.33,39.33,39.31,39.32,39.35,39.34,39.35]

psrgan_x2 = [39.57,39.86,40.78,41.28,41.67,41.61,42.06,42.18]
psrgan_x4 = [35.76,35.62,36.10,36.78,36.72,36.98,37.11,37.13]

hnct_x2 = [42.88,42.79,42.77,43.12,43.19,43.26,43.39,43.38]
hnct_x4 = [39.03,39.41,39.43,39.48,39.47,39.49,39.50,39.50]

plt.xlabel('x',fontsize=16)
plt.ylabel('y',fontsize=16)## FF9900

plt.plot(x_axis_data, imdn_x2, color='#FF9900',marker='.',linestyle='dashed', alpha=0.5, linewidth=2, label='IMDN')  # 'bo-'表示蓝色实线，数据点实心原点标注
## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
plt.plot(x_axis_data, dpsr_x2, 'go--', alpha=0.5, linewidth=2, label='DPSR')
plt.plot(x_axis_data, pan_x2, 'rs--', alpha=0.5, linewidth=2, label='PAN')
plt.plot(x_axis_data, psrgan_x2, 'c>--', alpha=0.5, linewidth=2, label='PSRGAN')
plt.plot(x_axis_data, swin_x2, 'm<--', alpha=0.5, linewidth=2, label='SwinIR')
plt.plot(x_axis_data, hnct_x2, 'yx--', alpha=0.5, linewidth=2, label='HNCT')
plt.plot(x_axis_data, y_axis_data_x2, 'b*--', alpha=0.5, linewidth=2, label='LISHN')

plt.legend()  # 显示上面的label

plt.xlabel('Number of training epochs')  # x_label
plt.ylabel('PSNR (dB)')  # y_label

# plt.yticks([35.0,35.5,36.0,36.5,37.0,37.5,38.0,38.5,39.0,39.5,40.0])
plt.yticks([39.0,39.5,40.0,40.5,41.0,41.5,42.0,42.5,43.0,43.5,44.0])
# 保存图片到本地
plt.savefig('x2-psnr.eps',dpi=600)
# plt.ylim(-1,1)#仅设置y轴坐标范围
plt.show()
