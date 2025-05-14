import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# 设置字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 项目任务和持续时间 (英文版)
tasks = [
    "Business Understanding", 
    "Data Understanding",
    "Data Preparation",
    "Modeling",
    "Evaluation",
    "Deployment",
    "Reporting & Release"
]

# 开始日期 (使用相对日期)
start_date = datetime.now()
month_duration = 30  # 以天为单位的月份近似值

# 计算每个任务的开始日期和结束日期
task_data = [
    # 任务名称, 开始月份(从0开始), 持续月数
    (tasks[0], 0, 1),
    (tasks[1], 0.5, 1.5),
    (tasks[2], 1.5, 2.5),
    (tasks[3], 2.5, 3),
    (tasks[4], 4.5, 1.5),
    (tasks[5], 3.5, 2.5),
    (tasks[6], 5, 1)
]

# 创建图形
fig, ax = plt.subplots(figsize=(12, 6))

# 添加任务栏
y_positions = np.arange(len(tasks))
for i, (task, start_month, duration) in enumerate(task_data):
    start = start_date + timedelta(days=start_month * month_duration)
    end = start + timedelta(days=duration * month_duration)
    
    # 绘制每个任务的条形图
    ax.barh(y_positions[i], (end - start).days, left=mdates.date2num(start), 
            height=0.5, align='center', 
            color='steelblue', alpha=0.8, 
            edgecolor='black')
    
    # 在条形图中添加任务名称
    midpoint = mdates.date2num(start) + (mdates.date2num(end) - mdates.date2num(start)) / 2
    ax.text(midpoint, y_positions[i], task, ha='center', va='center',
            color='white', fontweight='bold')

# 设置y轴标签
ax.set_yticks(y_positions)
ax.set_yticklabels(tasks)
ax.invert_yaxis()  # 从上到下显示任务

# 设置x轴为日期格式
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
fig.autofmt_xdate()  # 自动格式化日期标签

# 添加网格线
ax.grid(True, axis='x', linestyle='--', alpha=0.7)

# 添加标题和标签
ax.set_title('Project Gantt Chart', fontsize=16, pad=20)
ax.set_xlabel('Timeline', fontsize=12, labelpad=20)

# 添加月份标记
month_names = ['Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6']
month_positions = [start_date + timedelta(days=i*month_duration) for i in range(6)]
for i, pos in enumerate(month_positions):
    ax.axvline(x=mdates.date2num(pos), color='red', linestyle='--', alpha=0.3)
    ax.text(mdates.date2num(pos), len(tasks) + 0.5, month_names[i], ha='center')

# 保存图表
plt.tight_layout()
plt.savefig('project_gantt_chart_english.png', dpi=300)
plt.close()

print("Gantt chart has been created and saved as 'project_gantt_chart_english.png'") 