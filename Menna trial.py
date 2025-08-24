import cv2
import numpy as np
import matplotlib.pyplot as plt

ideal = cv2.imread("ideal.jpg")
gray_ideal = cv2.cvtColor(ideal, cv2.COLOR_BGR2GRAY)
_, binary1 = cv2.threshold(gray_ideal, 127, 255, cv2.THRESH_BINARY)
if np.sum(binary1 == 255) < np.sum(binary1 == 0):
    binary1 = cv2.bitwise_not(binary1)
kernel = np.ones((5, 5), np.uint8)
closed1 = cv2.morphologyEx(binary1, cv2.MORPH_CLOSE, kernel)
contours1_ideal, hier1 = cv2.findContours(closed1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
outer_ideal = max(contours1_ideal, key=cv2.contourArea)
idx_outer_1 = np.argmax([cv2.contourArea(c) for c in contours1_ideal])
inner_idxs_1 = [i for i, h in enumerate(hier1[0]) if h[3] == idx_outer_1]
inner_ideal = max([contours1_ideal[i] for i in inner_idxs_1], key=cv2.contourArea) if inner_idxs_1 else None
r_in_i = 0.0
if inner_ideal is not None:
    (_, _), r_in_i = cv2.minEnclosingCircle(inner_ideal)

samples = [
    ("sample2.jpg", "Sample 1"),
    ("sample3.jpg", "Sample 2"),
    ("sample4.jpg", "Sample 3"),
    ("sample5.jpg", "Sample 4"),
    ("sample6.jpg", "Sample 5"),
]

fig, axes = plt.subplots(2, 5, figsize=(22, 9))
axes[0, 2].imshow(cv2.cvtColor(ideal, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Ideal Gear", fontsize=12)
axes[0, 2].axis("off")
for j in [0, 1, 3, 4]:
    axes[0, j].axis("off")
for col, (sample_path, sample_name) in enumerate(samples):
    sample = cv2.imread(sample_path)
    if sample is None:
        print(f"⚠ Could not load {sample_path}")
        axes[1, col].axis("off")
        continue

    gray_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    _, binary2 = cv2.threshold(gray_sample, 127, 255, cv2.THRESH_BINARY)
    if np.sum(binary2 == 255) < np.sum(binary2 == 0):
        binary2 = cv2.bitwise_not(binary2)
    closed2 = cv2.morphologyEx(binary2, cv2.MORPH_CLOSE, kernel)
    contours2_sample, hier2 = cv2.findContours(closed2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    outer_sample = max(contours2_sample, key=cv2.contourArea)
    idx_outer_2 = np.argmax([cv2.contourArea(c) for c in contours2_sample])
    inner_idxs_2 = [k for k, h in enumerate(hier2[0]) if h[3] == idx_outer_2]
    inner_sample = max([contours2_sample[k] for k in inner_idxs_2], key=cv2.contourArea) if inner_idxs_2 else None

    defect_count, broken, worn, inner_defects = 0, 0, 0, 0
    vis = sample.copy()
    print(f"\n{sample_name}")

    if inner_ideal is not None and inner_sample is not None:
        (center_s, r_in_s) = cv2.minEnclosingCircle(inner_sample)
        (center_i, r_in_i) = cv2.minEnclosingCircle(inner_ideal)
        inner_diff = abs(2 * r_in_s - 2 * r_in_i)
        tolerance = 0.02 * (2 * r_in_i)  # 2%
        if inner_diff >= tolerance:
            defect_count += 1
            inner_defects += 1
            direction = "LARGER" if r_in_s > r_in_i else "SMALLER"
            print(f"Defect {defect_count}: Inner hole DIAMETER DIFFERENCE ({direction})")
            cv2.circle(vis, (int(center_s[0]), int(center_s[1])), int(r_in_s), (0, 255, 0), 2)
            cv2.putText(vis, "Inner defect", (int(center_s[0])-40, int(center_s[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    elif inner_ideal is not None and inner_sample is None:
        defect_count += 1
        inner_defects += 1
        print(f"Defect {defect_count}: Inner hole MISSING")
        (center_i, r_in_i) = cv2.minEnclosingCircle(inner_ideal)
        cv2.circle(vis, (int(center_i[0]), int(center_i[1])), int(r_in_i), (0, 255, 0), 2)
        cv2.putText(vis, "Inner missing", (int(center_i[0])-40, int(center_i[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    mask_ideal = np.ones_like(closed1)
    cv2.drawContours(mask_ideal, [outer_ideal], -1, 255, -1)
    if inner_ideal is not None:
        cv2.drawContours(mask_ideal, [inner_ideal], -1, 0, -1)
    mask_sample = np.ones_like(closed2)
    cv2.drawContours(mask_sample, [outer_sample], -1, 255, -1)
    if inner_sample is not None:
        cv2.drawContours(mask_sample, [inner_sample], -1, 0, -1)
    diff_mask = cv2.bitwise_xor(mask_ideal, mask_sample)
    contours_diff, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_diff:
        area = cv2.contourArea(cnt)
        if area <= 5:
            continue
        defect_count += 1
        if area > 400:
            broken += 1
            defect_type, color = "Broken tooth", (0, 0, 255)
        else:
            worn += 1
            defect_type, color = "Worn tooth", (255, 0, 0)
        cv2.drawContours(vis, [cnt], -1, color, 2)
        print(f"Defect {defect_count}: {defect_type}")

    print(f"Total defects: {defect_count}")
    print(f"Broken teeth: {broken}")
    print(f"Worn teeth: {worn}")
    print(f"Inner hole defects: {inner_defects}")

    axes[1, col].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[1, col].set_title(sample_name, fontsize=11)
    axes[1, col].axis("off")

plt.tight_layout()
plt.show()