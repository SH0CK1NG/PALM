import re
import numpy as np

# === 输入：原始多行 LaTeX 表格内容 ===
latex_rows = r"""
& & Random Label FT &91.13 & 45.05 & 75.99 
& 99.02 & 4.16 & 80.89 & 68.62 & 94.74 & 25.41 & 89.07 & 48.93 & 89.44 & 47.30 & 90.63 & 38.89 \\
& & GradAsc & 80.04 & 66.30 & 74.06
& 95.57 & 23.31 & 78.86 & 74.76 & 84.38 & 64.60 & 88.56 & 53.41 & 85.50 & 65.35 & 86.57 & 56.29 \\
& & \textbf{TFER (Ours)} & \textbf{89.64} & \textbf{51.75} & \textbf{76.18} 
& 99.11 & 4.04 & 78.28 & 77.96 & 93.83 & 32.05 & 82.96 & 74.76 & 89.80 & 44.41 & 88.80 & 46.64 \\
"""

# === 定义列趋势 ===
# ↑ 越大越好，↓ 越小越好
trend = ['up', 'down', 'up',    # Forget AUROC, Forget FPR95, Retain Acc
         'up', 'down',          # SVHN
         'up', 'down',          # Places365
         'up', 'down',          # LSUN
         'up', 'down',          # iSUN
         'up', 'down',          # DTD
         'up', 'down']          # AVG (OOD)

# === 解析行 ===
# 先合并续行（使用 \\ 作为行结束标记）
merged_lines = []
current_line = ""
for line in latex_rows.strip().splitlines():
    line = line.strip()
    if not line:
        continue
    
    # 追加到当前行
    if current_line:
        current_line += " " + line
    else:
        current_line = line
    
    # 如果以 \\ 结尾，说明这一行结束了
    if line.rstrip().endswith("\\\\") or line.rstrip().endswith("\\"):
        merged_lines.append(current_line)
        current_line = ""

# 添加最后一行（如果没有以 \\ 结尾）
if current_line:
    merged_lines.append(current_line)

# 解析合并后的行
rows = []
for line in merged_lines:
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", line)]
    name = re.search(r"&\s*&\s*([A-Za-z ()]+?)\s*&", line)
    name = name.group(1).strip() if name else ""
    rows.append((name, nums, line))

# === 转置比较 ===
cols = np.array([r[1] for r in rows])
best_mask = np.zeros_like(cols, dtype=bool)

for j, t in enumerate(trend):
    if t == 'up':
        best_mask[:, j] = cols[:, j] == np.nanmax(cols[:, j])
    else:
        best_mask[:, j] = cols[:, j] == np.nanmin(cols[:, j])

# === 生成带 \textbf 的新行 ===
new_rows = []
for (name, nums, raw), mask in zip(rows, best_mask):
    new_line = raw
    # 先移除所有现有的 \textbf{}，只保留数字
    new_line = re.sub(r"\\textbf\{([-+]?\d*\.\d+)\}", r"\1", new_line)
    
    # 找到所有数字的位置
    pattern = r"[-+]?\d*\.\d+"
    matches = list(re.finditer(pattern, new_line))
    
    # 从后往前替换，根据 mask 添加 \textbf
    for i in range(len(matches) - 1, -1, -1):
        if i >= len(nums) or i >= len(mask):
            continue
        match = matches[i]
        val = nums[i]
        matched_text = match.group(0)
        num_in_text = float(matched_text)
        
        # 检查是否是我们要处理的数字（浮点数比较）
        if abs(num_in_text - val) < 0.01:
            if mask[i]:
                replacement = f"\\textbf{{{val:.2f}}}"
            else:
                replacement = f"{val:.2f}"
            start, end = match.span()
            new_line = new_line[:start] + replacement + new_line[end:]
    
    new_rows.append(new_line)

print("\n".join(new_rows))
