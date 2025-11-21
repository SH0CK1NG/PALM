import re
import numpy as np

# === 输入：原始多行 LaTeX 表格内容 ===
latex_rows = r"""
& DER          & - & 79.06 & 79.06 & 79.06 & 83.65 & 81.06 & 78.89 & 76.18 & 87.20 & 64.04 & 86.04 & 71.64 & 80.68 & 76.56 & 73.90 & 83.29 \\
& iCaRL        & - & 79.91 & 79.91 & 79.91 & 83.38 & 79.12 & 78.78 & 77.01 & 87.29 & 65.61 & 87.79 & 67.43 & 81.07 & 75.59 & 72.95 & 83.66 \\
& LwF          & - & 79.76 & 79.76 & 79.76 & 83.24 & 79.64 & 78.67 & 77.32 & 87.36 & 65.45 & 88.02 & 67.16 & 81.11 & 75.71 & 73.06 & 83.68 \\
& EWC          & - & 79.58 & 79.58 & 79.58 & 82.48 & 79.62 & 77.36 & 81.42 & 83.84 & 77.88 & 87.86 & 66.01 & 82.08 & 76.76 & 76.34 & 82.72 \\
& Upper Bound  & - & 75.20 & 75.20 & 75.20 & 98.96 & 3.63 & 82.40 & 63.31 & 95.47 & 23.97 & 92.48 & 33.55 & 91.11 & 37.57 & 92.08 & 32.41 \\
& \textbf{PRCL (Ours)} &
- & \textbf{84.2} & \textbf{84.2} & \textbf{84.2} &
\textbf{99.60} & \textbf{2.18} &
\textbf{89.72} & \textbf{58.14} &
\textbf{97.86} & \textbf{12.39} &
\textbf{94.96} & \textbf{30.82} &
\textbf{96.51} & \textbf{18.26} &
\textbf{24.36} & \textbf{95.73}
\\
"""

# # === 定义列趋势 ===
# # ↑ 越大越好，↓ 越小越好
# trend = ['up', 'down', 'up',    # Forget AUROC, Forget FPR95, Retain Acc
#          'up', 'down',          # SVHN
#          'up', 'down',          # Places365
#          'up', 'down',          # LSUN
#          'up', 'down',          # iSUN
#          'up', 'down',          # DTD
#          'up', 'down']          # AVG (OOD)
# === 定义列趋势 ===
# ↑ 越大越好，↓ 越小越好
trend = ['up', 'up', 'up', 'up',    
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
