filename = "loss.txt"  # 替换为你的文件名

with open(filename, "r") as file:
    lines = file.readlines()

# 提取包含"epoch"字符串的行
epoch_lines = [line.strip() for line in lines if "Epoch" in line]

# 打印提取到的行
for line in epoch_lines:
    print(line)
