# 处理数据集
eneity_dict = {'no_eneity': 0}
rel_dict = {'no_relative': 0}


# 获取train中数据
with open("./train2id.txt", "r", encoding='utf8') as loader:
    for line in loader.readlines():
        if(len(line.strip().split())>1):
            h, t,r = line.strip().split()
            if h not in eneity_dict.keys():
                eneity_dict[h] = len(eneity_dict)
            if r not in rel_dict.keys():
                rel_dict[r] = len(rel_dict)
            if t not in eneity_dict.keys():
                eneity_dict[t] = len(eneity_dict)

with open("./test2id.txt", "r", encoding='utf8') as loader:
    for line in loader.readlines():
        if(len(line.strip().split())>1):
            h, t,r = line.strip().split()
            if h not in eneity_dict.keys():
                eneity_dict[h] = len(eneity_dict)
            if r not in rel_dict.keys():
                rel_dict[r] = len(rel_dict)
            if t not in eneity_dict.keys():
                eneity_dict[t] = len(eneity_dict)

with open("./valid2id.txt", "r", encoding='utf8') as loader:
    for line in loader.readlines():
        if(len(line.strip().split())>1):
            h, t,r = line.strip().split()
            if h not in eneity_dict.keys():
                eneity_dict[h] = len(eneity_dict)
            if r not in rel_dict.keys():
                rel_dict[r] = len(rel_dict)
            if t not in eneity_dict.keys():
                eneity_dict[t] = len(eneity_dict)

print(len(eneity_dict))
print(len(rel_dict))