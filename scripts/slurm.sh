#!/bin/bash

# 节点列表
nodes=("1pdijpn14trqa-0" "d4ofqnu7t5ab9-0" "fu5l6puknblo9-0")

# 提示用户输入节点编号
echo "请输入节点编号 (1-${#nodes[@]}):"
read n

# 检查输入是否有效
if [[ $n -lt 1 || $n -gt ${#nodes[@]} ]]; then
  echo "无效的节点编号。请输入一个介于 1 和 ${#nodes[@]} 之间的数字。"
  exit 1
fi

# 获取对应的节点
node=${nodes[$((n-1))]}

# 创建临时 SLURM 脚本
slurm_script=$(mktemp /tmp/slurm_script.XXXXXX)

cat <<EOT > $slurm_script
#!/bin/bash
#SBATCH --job-name=compute_embeddings
#SBATCH --output=output_%j.txt
#SBATCH --nodelist=$node

# 在选定的节点上执行命令
sh scripts/computing_embeddings.sh
EOT

# 提交 SLURM 作业
sbatch $slurm_script

# 删除临时脚本
rm $slurm_script
