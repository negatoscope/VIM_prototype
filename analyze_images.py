import os
import csv
from PIL import Image
import numpy as np
import colorsys

# Paths
input_folder = "C:/Users/Luis/Desktop/VIM_prototype/images/originals"
output_csv = "C:/Users/Luis/Desktop/VIM_prototype/metrics_normalized.csv"

# Prepare CSV
with open(output_csv, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "Filename", 
        "Luminance_mean", "Luminance_std",
        "Contrast_mean", "Contrast_std",
        "Saturation_mean", "Saturation_std"
    ])
    
    # Loop through images
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
            continue
        filepath = os.path.join(input_folder, filename)
        
        # Load image
        img = Image.open(filepath).convert("RGB")
        arr = np.array(img, dtype=float) / 255.0  # normalize RGB to 0-1
        
        # --- Luminance ---
        lum = 0.2126*arr[:,:,0] + 0.7152*arr[:,:,1] + 0.0722*arr[:,:,2]
        mean_lum = lum.mean()        # normalized 0-1
        std_lum = lum.std()          # normalized 0-1
        
        # --- RMS Contrast (std of luminance) ---
        contrast_mean = lum.mean()   # same as luminance mean
        contrast_std = lum.std()     # same as luminance std (RMS contrast)
        
        # --- Saturation ---
        h, l, s = np.vectorize(colorsys.rgb_to_hls)(arr[:,:,0], arr[:,:,1], arr[:,:,2])
        mean_saturation = s.mean()
        std_saturation = s.std()
        
        # Save row
        writer.writerow([
            filename,
            round(mean_lum, 4), round(std_lum, 4),
            round(contrast_mean, 4), round(contrast_std, 4),
            round(mean_saturation, 4), round(std_saturation, 4)
        ])
        
        print(f"{filename} => Luminance: {mean_lum:.3f}±{std_lum:.3f}, "
              f"Contrast: {contrast_mean:.3f}±{contrast_std:.3f}, "
              f"Saturation: {mean_saturation:.3f}±{std_saturation:.3f}")

print(f"Analysis complete. Results saved to {output_csv}")
