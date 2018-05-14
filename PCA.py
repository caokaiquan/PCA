# #PCA
# PCA是一种常用的数据降维
#
# A与B的内积：A⋅B=|A||B|cos(a)
# A与B的内积等于A到B的投影长度乘以B的模。如果我们假设B的模为1，即让
# ，那么就变成了：
#                   A⋅B=|A|cos(a)
# 也就是说，设向量B的模为1，则A与B的内积值等于A向B所在直线投影的矢量长度！这就是内积的一种几何解释。
#
# 我们这里假设基是正交的（即内积为0，或直观说相互垂直），但可以成为一组基的唯一要求就是线性无关，非正交的基也是可以的。不过因为正交基有较好的性质，所以一般使用的基都是正交的。
# 若我们想求出一点到新基上的坐标，我们只需要求出该点与新基的内积
#
# 对于需要转换的记录，我们将它们表示成矩阵形式，其中每一列为一条数据记录，而一行为一个字段。为了后续处理方便，我们首先将每个字段内所有值都减去字段均值，其结果是将每个字段都变为均值为0（这样做的道理和好处后面会看到）。
#
# PCA的核心问题就是我们需要找到一组基，投射在该组基的数据尽可能的保留最多的信息，一种直观的看法是：希望投影后的投影值尽可能分散，这种分散程度，可以用数学上的方差来表述。此处，一个字段的方差可以看做是每个元素与字段均值的差的平方和的均值。由于上面我们已经将每个字段的均值都化为0了，因此方差可以直接用每个元素的平方和除以元素个数表示，于是上面的问题被形式化表述为：寻找一组基，使得所有数据变换为这个基上的坐标表示后，方差值最大。
#
# 对于降维到更高维，而不只是一维，如果我们还是单纯只选择方差最大的方向，很明显，这个方向与第一个方向应该是“几乎重合在一起”，显然这样的维度是没有用的，因此，应该有其他约束条件。从直观上说，让两个字段尽可能表示更多的原始信息，我们是不希望它们之间存在（线性）相关性的，因为相关性意味着两个字段不是完全独立，必然存在重复表示的信息。如果我们还是单纯只选择方差最大的方向，很明显，这个方向与第一个方向应该是“几乎重合在一起”，显然这样的维度是没有用的，因此，应该有其他约束条件。从直观上说，让两个字段尽可能表示更多的原始信息，我们是不希望它们之间存在（线性）相关性的，因为相关性意味着两个字段不是完全独立，必然存在重复表示的信息。数学上可以用两个字段的协方差表示其相关性
# 当协方差为0时，表示两个字段完全独立。为了让协方差为0，我们选择第二个基时只能在与第一个基正交的方向上选择。因此最终选择的两个方向一定是正交的。
#
# 至此，我们得到了降维问题的优化目标：将一组N维向量降为K维（K大于0，小于N），其目标是选择K个单位（模为1）正交基，使得原始数据变换到这组基上后，各字段两两间协方差为0，而字段的方差则尽可能大（在正交的约束下，取最大的K个方差）。
#
# （1/m)*X*X.T为协方差矩阵，X为以一列组成一条记录的矩阵，这样协方差矩阵就把方差与协方差统一到一个矩阵上，它是一个对称矩阵，对角线为各个字段的方差，以外为协方差。
# 我们发现要达到优化目前，等价于将协方差矩阵对角化：即除对角线外的其它元素化为0，并且在对角线上将元素按大小从上到下排列，这样我们就达到了优化目的。
#
#
# 设原始数据矩阵X对应的协方差矩阵为C，而P是一组基按行组成的矩阵，设Y=PX，则Y为X对P做基变换后的数据。设Y的协方差矩阵为D，我们推导一下D与C的关系：
# D = (1/m) * Y * Y.T
#   = (1/m) * PX * (PX).T
#   = (1/m) * PX * X.T * P.T
#   = P * (1/m)* X * X.T * P.T
#   = P * C * P.T
# 现在事情很明白了！我们要找的P不是别的，而是能让原始协方差矩阵对角化的P。换句话说，优化目标变成了寻找一个矩阵P，满足P*C*P.T是一个对角矩阵，并且对角元素按从大到小依次排列，那么P的前K行就是要寻找的基，用P的前K行组成的矩阵乘以X就使得X从N维降到了K维并满足上述优化条件。
# 协方差矩阵C是一个是对称矩阵，在线性代数上，实对称矩阵有一系列非常好的性质：
#
# 1）实对称矩阵不同特征值对应的特征向量必然正交。
#
# 2）设特征向量重数为r，则必然存在r个线性无关的特征向量对应于，因此可以将这r个特征向量单位正交化。
#
# 由上面两条可知，一个n行n列的实对称矩阵一定可以找到n个单位正交特征向量(这句话最核心)，设这n个特征向量为，我们将其按列组成矩阵：
#
# 则对协方差矩阵C有如下结论：
#
# 其中为对角矩阵，其对角元素为各特征向量对应的特征值（可能有重复）。




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('iris.data')
df.head()

df.columns = ['sepal_len','sepal_wid','petal_len','petal_wid','class']
df.head()

x = df.ix[:,0:4].values
y = df.ix[;,4].values

from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(x)
print(X_std)

mean_vec = np.mean(X_std,axis = 0)
cov_mat = (X_std - mean_vec).T.dot((X_std-mean_vec)) / (X_std.shape[0] - 1)
print('covariance matrix \n%s' %cov_mat)

print('numpy covariance matrix:\n%s' %np.cov(X_std.T))

cov_mat = np.cov(X_std.T)
eig_vals,eig_vecs = np.linalg.eig(cov_mat)
print('eigenvectors \n%s' %eig_vecs)
print('eigenvalues \n%s' %eig_vals)

eig_pairs = [(np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]
print(eig_pairs)
print('---------------------------------------------------------------------------')
eig_pairs.sort(key = lambda x: x[0],reverse=True)

print('eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])



tot = sum(eig_vals)
var_exp = [(i/tot)*100 for i in sorted(eig_vals,reverse = True)]
print(var_exp)
cum_var_exp = np.cumsum(var_exp)
print(cum_var_exp)



plt.figure(figsize=(6,4))
plt.bar(list(range(4)),var_exp,alpha = 0.5,align = 'center',label = 'individual explained variance')
plt.step(list(range(4)),cum_var_exp,where = 'mid',label = 'cumulative explained variance')
plt.ylabel('explained variance ratio')
plt.xlabel('principal components')
plt.legend(loc = 'best')
plt.tight_layout()
plt.show()


matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),eig_pairs[1][1].reshape(4,1)))
print('Matrix W:\n',matrix_w)

Y = X_std.dot(matrix_w)
print(Y)



































