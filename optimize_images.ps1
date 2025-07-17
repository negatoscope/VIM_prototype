# --- Image Optimization Script for VIM Experiment (PowerShell) ---

# --- Configuration ---
# PASTE THE FULL PATH to your gmic.exe file here, inside the quotes.
$GMIC_PATH = "C:\gmic\gmic\gmic\gmic.exe" # <-- PASTE YOUR PATH HERE

# --- Configuration ---
# 1. Set the desired JPEG quality (85-95 is a good range).
$QUALITY = 88

# 2. Set the maximum width/height for the images.
$MAX_DIMENSION = 1920
# --- End Configuration ---


# Get the main 'images' directory
$mainImageDir = Get-Item -Path ".\images"

# Find all subdirectories inside 'images'
Get-ChildItem -Path $mainImageDir.FullName -Directory | ForEach-Object {
    $dir = $_
    Write-Host "--- Processing directory: $($dir.FullName) ---" -ForegroundColor Green

    # Create a temporary output folder
    $tempOutputDir = Join-Path -Path $dir.FullName -ChildPath "temp_output"
    if (-not (Test-Path $tempOutputDir)) {
        New-Item -ItemType Directory -Path $tempOutputDir | Out-Null
    }

    # Loop through every PNG file in the directory
    Get-ChildItem -Path $dir.FullName -Filter *.png | ForEach-Object {
        $file = $_
        $filename = $file.BaseName
        
        Write-Host "Optimizing: $($file.Name)"

        # Use G'MIC to resize and convert the image
        # Note: 'gmic' must be in your system's PATH
        & $GMIC_PATH $file.FullName -resize2dx 1920 -remove_opacity -output "$($tempOutputDir)\$($filename).jpg",$QUALITY, interleave=1 -quit
    }

    # Replace the old PNGs with the new JPGs
    Write-Host "Replacing original files in $($dir.Name)..."
    Remove-Item -Path "$($dir.FullName)\*.png" -Force
    Move-Item -Path "$($tempOutputDir)\*" -Destination $dir.FullName
    Remove-Item -Path $tempOutputDir -Force
}

Write-Host "--- All image optimization complete! ---" -ForegroundColor Green
Write-Host "IMPORTANT: Remember to change 'imageExtension' to '.jpg' in your index.html script."