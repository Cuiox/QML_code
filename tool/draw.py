import numpy as np
import matplotlib.pyplot as plt

def draw_plt():
        
    filename = "acc.txt"  # 替换为你的文件名
    fig_name = "acc"
    title = "accuracy for each epoch"
    xlabel = "epoch"
    ylabel = "accuracy "
    data = np.loadtxt(filename)  # 读取txt文件中的数据
    
    #data = data / 200
    
    fig = plt.figure()
    # 绘制折线图
    plt.plot(data)

    # 添加标题和标签
    plt.title(f"{title}")
    plt.xlabel(f"{xlabel}")
    plt.ylabel(f"{ylabel}")

    # 显示图形
    #plt.show()
    fig.savefig(f"{fig_name}.png")
    
    plt.close()
    print(f"finish: save in {fig_name}.png")

def read_txt():
    filename = "acc.txt"  # 替换为你的文件名
    fig_name = "acc_num"
    title = "accuracy for each epoch"
    xlabel = "epoch"
    ylabel = "accuracy "
    data = np.loadtxt(filename)  # 读取txt文件中的数据
    print(data)
    data = data / 200
    print(data)
    
def read_csv():
    filename = "test.csv"  # 替换为你的文件名

    data = np.loadtxt(filename, delimiter=",")

    num_qubit = 6
    
    # 按列保存数据
    loss = data[:, 0]
    acc = data[:, 1] / 200
    if 0:
        p = []
        for i in range(6):
            p.append(data[:, 2+i])
        #column3 = data[:, 2]
        #column8 = data[:, 7]
        print(p)

    p = data[:, -num_qubit:] # # 获取最后6列数据

    mean = np.mean(p, axis=1)
    std = np.std(p, axis=1)
    dist = np.linalg.norm(p, ord=2, axis=1)
    var = np.var(p, axis=1)
    cv = std / mean
      
    if 1:
        # 按列保存平均值和标准差
        results = np.column_stack((
            loss,
            acc,
            mean,
            std,
            dist,
            var,
            cv,
        ))

        # 保存结果到CSV文件
        np.savetxt("results.csv", results, delimiter=",")
        
        print(f"finish: save in results.csv")
        

    
if __name__ == "__main__":
    #draw_plt()
    #read()
    read_csv()