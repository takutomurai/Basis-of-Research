import numpy as np
import os
import matplotlib.pyplot as plt
from MIP import maximum_intensity_projection

# ベースディレクトリ
base_dir = r"C:\Users\murai\research data"
# 出力ディレクトリ
output_dir = r"C:\Users\murai\output_mip_trial"
os.makedirs(output_dir, exist_ok=True)

# MIC (1) から MIC (10) までループ
for i in range(1, 11):
    folder_name = f"MIC ({i})"
    file_path = os.path.join(base_dir, folder_name, "image.npy")
    if os.path.exists(file_path):
        volume = np.load(file_path)
        mip_coronal = maximum_intensity_projection(volume, axis=1)
        # 画像ファイルとして保存
        output_path = os.path.join(output_dir, f"coronal_mip_MIC_{i}.png")
        plt.imsave(output_path, mip_coronal, cmap='gray')
        print(f"保存しました: {output_path}")
    else:
        print(f"ファイルが見つかりません: {file_path}")