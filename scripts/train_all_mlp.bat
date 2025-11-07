@echo off
REM Train all three MLP autoencoders sequentially

echo ================================
echo Training MLP Autoencoder - PSD
echo ================================
python scripts/train_mlp_autoencoder.py ^
    --input data/mock_lightcurves/preprocessed.h5 ^
    --channel psd ^
    --output_dir models/mlp ^
    --epochs 50 ^
    --batch_size 128 ^
    --lr 1e-3 ^
    --mask_ratio 0.7 ^
    --block_size 50

if errorlevel 1 (
    echo Error training PSD model
    exit /b 1
)

echo.
echo ================================
echo Training MLP Autoencoder - ACF
echo ================================
python scripts/train_mlp_autoencoder.py ^
    --input data/mock_lightcurves/preprocessed.h5 ^
    --channel acf ^
    --output_dir models/mlp ^
    --epochs 50 ^
    --batch_size 128 ^
    --lr 1e-3 ^
    --mask_ratio 0.7 ^
    --block_size 50

if errorlevel 1 (
    echo Error training ACF model
    exit /b 1
)

echo.
echo ================================
echo Training MLP Autoencoder - F-stat
echo ================================
python scripts/train_mlp_autoencoder.py ^
    --input data/mock_lightcurves/preprocessed.h5 ^
    --channel fstat ^
    --output_dir models/mlp ^
    --epochs 50 ^
    --batch_size 128 ^
    --lr 1e-3 ^
    --mask_ratio 0.7 ^
    --block_size 50

if errorlevel 1 (
    echo Error training F-stat model
    exit /b 1
)

echo.
echo ================================
echo All models trained successfully!
echo ================================
