#!/bin/bash

# Face Enhancement Models Installation Script
# This script downloads the required models for ultra-realistic face enhancement

set -e

echo "🎨 Face Enhancement Models Installer"
echo "===================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create cache directories
echo -e "${BLUE}Creating cache directories...${NC}"
mkdir -p ~/.cache/gfpgan
mkdir -p ~/.cache/realesrgan
mkdir -p ~/.cache/codeformer

# Download GFPGAN v1.4 (required for high quality and ultra realistic modes)
echo -e "\n${BLUE}Downloading GFPGAN v1.4 model...${NC}"
if [ -f ~/.cache/gfpgan/GFPGANv1.4.pth ]; then
    echo -e "${GREEN}✓ GFPGAN model already exists${NC}"
else
    echo "Downloading from TencentARC/GFPGAN..."
    curl -L -o ~/.cache/gfpgan/GFPGANv1.4.pth \
        https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
    echo -e "${GREEN}✓ GFPGAN v1.4 downloaded successfully${NC}"
fi

# Download Real-ESRGAN x4plus (optional, for ultra realistic mode)
echo -e "\n${BLUE}Downloading Real-ESRGAN x4plus model (optional)...${NC}"
if [ -f ~/.cache/realesrgan/RealESRGAN_x4plus.pth ]; then
    echo -e "${GREEN}✓ Real-ESRGAN model already exists${NC}"
else
    echo "Downloading from xinntao/Real-ESRGAN..."
    curl -L -o ~/.cache/realesrgan/RealESRGAN_x4plus.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
    echo -e "${GREEN}✓ Real-ESRGAN x4plus downloaded successfully${NC}"
fi

# Download detection and alignment models (used by GFPGAN)
echo -e "\n${BLUE}Downloading face detection models...${NC}"
mkdir -p ~/.cache/gfpgan/weights
if [ -f ~/.cache/gfpgan/weights/detection_Resnet50_Final.pth ]; then
    echo -e "${GREEN}✓ Detection model already exists${NC}"
else
    curl -L -o ~/.cache/gfpgan/weights/detection_Resnet50_Final.pth \
        https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth
    echo -e "${GREEN}✓ Detection model downloaded${NC}"
fi

if [ -f ~/.cache/gfpgan/weights/parsing_parsenet.pth ]; then
    echo -e "${GREEN}✓ Parsing model already exists${NC}"
else
    curl -L -o ~/.cache/gfpgan/weights/parsing_parsenet.pth \
        https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth
    echo -e "${GREEN}✓ Parsing model downloaded${NC}"
fi

# Verify installations
echo -e "\n${BLUE}Verifying installations...${NC}"
echo "Model locations:"
echo "  GFPGAN: ~/.cache/gfpgan/GFPGANv1.4.pth"
echo "  Real-ESRGAN: ~/.cache/realesrgan/RealESRGAN_x4plus.pth"

# Calculate total size
echo -e "\n${BLUE}Disk space used:${NC}"
du -sh ~/.cache/gfpgan ~/.cache/realesrgan 2>/dev/null || true

echo -e "\n${GREEN}✓ All models installed successfully!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Make sure you've installed Python dependencies:"
echo "   pip install -r backend/requirements.txt"
echo ""
echo "2. Restart your backend server:"
echo "   cd backend && python app.py"
echo ""
echo "3. The enhancement pipeline will now automatically use these models"
echo ""
echo -e "${GREEN}Enhancement modes available:${NC}"
echo "  • fast: No enhancement (fastest)"
echo "  • studio/high_quality: GFPGAN enhancement (balanced)"
echo "  • cinematic/ultra_realistic: Full pipeline (best quality)"
echo ""
