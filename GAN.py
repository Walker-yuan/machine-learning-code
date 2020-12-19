import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

# Hyper-parameters
latent_size = 100
hidden_size = 256
image_size = 28*28
num_epochs = 100
batch_size = 100
learning_rate = 0.0002

DOWNLOAD_MNIST = False

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,),  # 3 for RGB channels
                         std=(0.5,))])

# 数据集下载(导入)
mnist = torchvision.datasets.MNIST(root='./data/',
                                   train=True,
                                   transform=transform,
                                   download=DOWNLOAD_MNIST)


# 数据集封装
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size,
                                          shuffle=True)

# 判别器, 其作用是尽量区分其输入的图片是真实图片还是噪声通过生成器生成的虚假图片
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),                        ## 激活函数
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),                ## 判断真假, 相当于做二分类, 真实判为1, 虚假判为0
    nn.Sigmoid())

# 生成器, 通过噪声生成图片, 其作用是生成的图片尽量骗过判别器, 即生成的图片尽量看起来比较真实, 或者说达到了以假乱真的地步
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh()
)



# 定义损失函数以及优化器
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate)
g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate)



def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

## 将梯度归为0
def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


# 用于保存每个epoch的损失和得分
d_losses = np.zeros(num_epochs)
g_losses = np.zeros(num_epochs)
real_scores = np.zeros(num_epochs)
fake_scores = np.zeros(num_epochs)

# 训练
total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):

        ## 生成虚假图片
        images = images.view(batch_size, -1).float()
        images = Variable(images)

        # 生成真实图片和虚假图片的类标, 真实图片类标为1, 虚假图片的类标为0
        real_labels = torch.ones(batch_size, 1)
        real_labels = Variable(real_labels)
        fake_labels = torch.zeros(batch_size, 1)
        fake_labels = Variable(fake_labels)


        ## 固定生成器, 训练判别器
        # 计算判别器中真实图片的交叉熵损失：BCE_Los(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        # 计算判别器中虚假图片的交叉熵损失
        z = torch.randn(batch_size, latent_size)
        z = Variable(z)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        # 计算判别器的损失
        d_loss = d_loss_real + d_loss_fake
        ## 梯度归0
        reset_grad()
        ## 损失逆向传播
        d_loss.backward()
        ## 更新判别器参数
        d_optimizer.step()

        ## 固定判别器, 训练生成器

        # 计算生成器的损失
        z = torch.randn(batch_size, latent_size)
        z = Variable(z)
        fake_images = G(z)
        outputs = D(fake_images)

        # 最大化 Elog(D(G(z))
        # https://arxiv.org/pdf/1406.2661.pdf
        g_loss = criterion(outputs, real_labels)

        ## 梯度归0, 损失逆向传播, 更新生成器参数
        reset_grad()
        g_loss.backward()
        g_optimizer.step()


        ## 保存每个epoch的损失和得分
        d_losses[epoch] = d_loss.data
        g_losses[epoch] = g_loss.data
        real_scores[epoch] = real_score.mean().data
        fake_scores[epoch] = fake_score.mean().data

        ## 每两百步打印一次结果
        if (i + 1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                  .format(epoch, num_epochs, i + 1, total_step, d_loss.data, g_loss.data,
                          real_score.mean().data, fake_score.mean().data))


   ## 利用训练的生成器来生成图片并保存
    r, c = 5, 5
    noise = torch.randn(r * c, latent_size)
    gen_imgs = G(noise)
    # Rescale images 0 - 1
    gen_imgs = denorm(gen_imgs)

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :].detach().view(28, 28), cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("images/%d.png" % epoch) ## 保存的路径
    plt.close()