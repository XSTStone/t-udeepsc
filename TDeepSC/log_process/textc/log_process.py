import re
import matplotlib.pyplot as plt

# 日志文件路径
log_file_path = '/home/local/Stone/code/t-udeepsc/TDeepSC/log/textc/2024_10_15_1336_training.log'

# 存储提取的值
epochs = []
losses = []
accs = []
lrs = []

# 读取日志文件并提取值
with open(log_file_path, 'r') as file:
    for line in file:
        match = re.search(r'\[Epoch: (\d+), Batch: \d+/\d+, Loss: ([\d.e-]+), Acc: ([\d.]+), lr: ([\d.e-]+)', line)
        if match:
            epoch = int(match.group(1))
            loss = match.group(2)
            acc = match.group(3)
            lr = match.group(4)
            
            try:
                # 转换为浮点数，并添加到列表
                epochs.append(epoch)
                losses.append(float(loss))
                accs.append(float(acc))
                lrs.append(float(lr))
            except ValueError as e:
                print(f"Skipping line due to conversion error: {line.strip()}")
                print(e)

# 绘制 Loss 图
plt.figure()
plt.plot(epochs, losses, label='Loss', color='blue')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig('/home/local/Stone/code/t-udeepsc/TDeepSC/log/textc/loss_plot.png')  # 保存图表
plt.close()

# 绘制 Acc 图
plt.figure()
plt.plot(epochs, accs, label='Accuracy', color='orange')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.savefig('/home/local/Stone/code/t-udeepsc/TDeepSC/log/textc/accuracy_plot.png')  # 保存图表
plt.close()

# 绘制 lr 图
plt.figure()
plt.plot(epochs, lrs, label='Learning Rate', color='green')
plt.title('Learning Rate over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.legend()
plt.grid()
plt.savefig('/home/local/Stone/code/t-udeepsc/TDeepSC/log/textc/learning_rate_plot.png')  # 保存图表
plt.close()

