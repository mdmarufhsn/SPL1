# SPL1
# SPL face / non-face classifier

This is the first C++ implementation of the project pipeline:

```text
JPG/JPEG -> grayscale pixels -> non-overlapping tiles
         -> 256-bin LBP histogram per tile
         -> coordinate-wise max pooling -> Random Forest -> FACE / NON-FACE
```

It uses Windows GDI+ to decode JPEGs and does not use OpenCV. LBP and the Random Forest are implemented in this project. Each input image becomes one labeled training example after its tile histograms are max-pooled. The predicted label is therefore for the whole input image.

## Build

On Windows with MinGW g++:

```powershell
.\build.bat
```

Or use CMake with a Windows C++ compiler:

```powershell
cmake -S . -B build
cmake --build build --config Release
```

GDI+ is a Windows system component. The build links `gdiplus` and `gdi32`; OpenCV and Python are not used.

## Dataset layout

Place training images in these folders. Nested subfolders are also scanned.

```text
dataset/
  face/
    face_001.jpg
    ...
  non_face/
    scene_001.jpeg
    ...
```

The training set must contain at least one image from each class. For evaluation, use the same folder layout under a separate test directory. Keep test images separate from training images; the `evaluate` command does not create a split automatically.

## Commands

Export every tile's feature vector and the pooled vector for one image:

```powershell
.\spl_rf.exe extract sample.jpg features.csv 64
```

Train and save a forest. Defaults are 100 trees, 64-pixel tiles, and Information Gain (entropy reduction) for tree splits. Supported split criteria are `information_gain`, `gini`, and `hellinger`.

```powershell
.\spl_rf.exe train dataset face_model.txt 100 64 information_gain
```

Predict one JPG/JPEG image:

```powershell
.\spl_rf.exe predict face_model.txt sample.jpg
```

Evaluate a saved model on a separate labeled test set:

```powershell
.\spl_rf.exe evaluate face_model.txt test_dataset
```

## Current implementation choices

- Tiles do not overlap. The default tile is 64 by 64 pixels; edge tiles keep their remaining pixels instead of being dropped or padded.
- Each tile gets a normalized 256-bin LBP histogram using the eight immediate neighbors around each center pixel. Very small edge tiles produce an all-zero histogram.
- Max pooling takes the largest value at each of the 256 histogram positions across all tiles in one image.
- Each Random Forest tree trains on a bootstrap sample. At each node it checks a random subset of features (square root of the feature count by default), searches thresholds between observed values, and votes by leaf class. The default maximum depth is 12.
- `information_gain` uses entropy reduction. `hellinger` uses the Hellinger distance between the two class-conditional left/right distributions and can be selected for imbalanced classes.
- The test evaluator reports accuracy, precision, recall, F1, and a confusion matrix. It evaluates the supplied test folder as-is; it does not train on it.

This is an initial, explainable implementation. The tile size, number of trees, dataset split, and preferred split criterion can be adjusted when your teacher confirms the exact settings.
