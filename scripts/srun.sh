#!/bin/bash
# srun --nodelist=fu5l6puknblo9-0 --time=00:10:00 --pty /bin/bash
# 提示用户输入节点编号
echo "请输入节点编号 (1-3):"
read n

# 节点列表
nodes=("ceo9vk5s9pvki-0" "d4ofqnu7t5ab9-0" "ddjq1d46n9m6b-0")

# 检查输入是否有效
if [[ $n -lt 1 || $n -gt ${#nodes[@]} ]]; then
  echo "无效的节点编号。请输入一个介于 1 和 ${#nodes[@]} 之间的数字。"
  exit 1
fi

# 获取对应的节点
node=${nodes[$((n-1))]}
srun --nodelist=$node --time=10:00:00 --pty /bin/bash
