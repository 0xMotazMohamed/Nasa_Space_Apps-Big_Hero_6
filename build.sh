#!/usr/bin/env bash
set -e

echo "🚀 Starting custom Render build..."

# 1️⃣ Use stable pip that skips strict dependency checks
pip install --upgrade "pip<24" setuptools wheel

# 2️⃣ Install dependencies in controlled order
pip install numpy==2.3.3
pip install cartopy==0.25.0 dask==2025.9.1 earthaccess==0.15.1 matplotlib==3.10.6 \
            xarray==2025.9.1 keyring==21.8.0 fastapi==0.115.4 uvicorn==0.32.0 \
            mangum==0.17.0 truststore

# 3️⃣ Install arcgis without dependency enforcement
pip install arcgis==2.4.1.3  --force-reinstall

# 4️⃣ Verify installed versions
python - <<EOF
import numpy, arcgis
print("✅ NumPy version:", numpy.__version__)
print("✅ ArcGIS version:", arcgis.__version__)
EOF

echo "🎯 Build finished successfully!"
