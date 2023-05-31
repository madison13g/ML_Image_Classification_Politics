import numpy as np
import matplotlib.pyplot as plt

#loading numpy arrays
lenet = np.load('./lenet5.npy')
densenet = np.load('./densenet.npy')
custom = np.load('./custom.npy')

#producing loss plot 
plt.plot(lenet, label='LeNet-5')
plt.plot(densenet, label='DenseNet')
plt.plot(custom, label='Custom')
plt.legend(frameon=False)
plt.xlabel('Num. of Epochs')
plt.savefig('./training_cost.png', dpi=200)
plt.clf()


#loading numpy arrays
lenet = np.load('./lenet5_valid.npy')
densenet = np.load('./densenet_valid.npy')
custom = np.load('./custom_valid.npy')

#producing validation error plot 
plt.plot(lenet, label='LeNet-5')
plt.plot(densenet, label='DenseNet')
plt.plot(custom, label='Custom')
plt.legend(frameon=False)
plt.xlabel('Num. of Epochs')
plt.savefig('./validation_error.png', dpi=200)
plt.clf()


#producing barplot
data = {'LeNet-5':60.21, 'DenseNet':51.76, 'Custom':51.41}
courses = list(data.keys())
values = list(data.values())
fig, ax = plt.subplots()
#plt.bar(courses, values, color ='maroon', width = 0.4)
plt.xlabel("Architecture")
plt.ylabel("Test Error (%)")
x = np.arange(len(courses)) 
ax.set_xticks(x)
ax.set_xticklabels(courses)
plt.ylim((0,65))
pps = ax.bar(x - (0.4/2)+0.2, values, 0.4, label='population', color='maroon')
for p in pps:
   height = p.get_height()
   ax.annotate('{}'.format(height),
      xy=(p.get_x() + p.get_width() / 2, height),
      xytext=(0, 3), # 3 points vertical offset
      textcoords="offset points",
      ha='center', va='bottom')
plt.savefig('./finaltest_error.png', dpi=200)
