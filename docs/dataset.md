# Dataset: INbreast (Kaggle copy)

Findings from `notebooks/01_data_exploration.ipynb` on the raw data in `data/raw/INbreast Release 1.0/`. Source: [ramanathansp20/inbreast-dataset](https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset) on Kaggle, which repackages the original INbreast Release 1.0 (Breast Research Group, INESC Porto).

**Scope decision:** the project segments **masses only**. Calcifications are a different problem (see below) and are left for later.

## License and use

The Kaggle page lists the license as "Unknown" and has no description. The original release was distributed for research use on request. Treat the data as not redistributable: no samples in the repository, tests or docs, and keep `data/` git-ignored.

## Contents

| Folder / file | Content |
|---|---|
| `AllDICOMs/` | 410 mammograms (`.dcm`), plus two MATLAB helper scripts |
| `AllXML/`, `AllROI/` | Lesion annotations for 343 images, the same annotations in two formats (OsiriX `.xml` and `.roi`) |
| `PectoralMuscle/` | Pectoral muscle outlines for 201 images |
| `MedicalReports/` | 117 text reports, named by patient id |
| `INbreast.csv` / `.xls` | Per-image summary: laterality, view, date, ACR density, BI-RADS |
| `inbreast.pdf` | The INbreast publication |

## Images

- 410 images, **108 patients** (the publication speaks of 115 cases). 99 patients have one exam and 9 have two (distinct acquisition dates).
- The filename is `<file id>_<patient id>_MG_<laterality>_<view>_ANON`. The patient id is the grouping key for splits (confirmed by `README.txt`). `INbreast.csv` has the patient id removed and links by file id only. Its file ids, laterality and view match the DICOM filenames for all 410 images.
- Views: 203 CC, 206 MLO and one right-breast image with view `FB`. Left/right is balanced (209 / 201). See [Views](#views) below.
- Two sizes: 3328 x 2560 (243 images) and 4084 x 3328 (167 images). Both occur in every acquisition period.
- DICOM headers: 16-bit unsigned, `MONOCHROME2` (higher value = brighter, no inversion needed), uncompressed, modality `OT`. There is **no `PixelSpacing`** in the headers; if physical size is needed it has to come from the publication.
- Actual intensities use 12 bits: the maximum value is at most 4095 in every image, although 16 bits are stored.
- Orientation follows laterality: left-breast images have the chest wall on the left, right-breast images on the right (checked on all 410). Flip one side to a common orientation before training.
- BI-RADS (per image): 1: 67, 2: 220, 3: 23, 4a: 13, 4b: 8, 4c: 22, 5: 49, 6: 8.

### Views

- **`ML` is taken to be MLO.** The filenames write `ML` and `INbreast.csv` writes `MLO` for the same 206 images. The paper lists only CC and MLO views, and the DICOM headers carry no view information (no `ViewPosition`, positioner angles or `PatientOrientation` values) to confirm it. A true lateral view (90 degrees) would be a different view, but nothing in the dataset describes one. The manifest build checks that the filename and the CSV agree.
- **`FB` is its own view and is kept in the manifest, but excluded from training and evaluation.** `FB` means "from below" (caudocranial), a reversed CC taken for patients where the standard CC is hard to position. Only one image has it (`50994300`, a right breast, 2009-01 exam). The same breast has two CC images and an MLO in that exam with an annotated mass (BI-RADS 4a, a 12 mm nodule according to the report), but the `FB` image has no mass annotation, only one calcification point, and its framing differs (less of the breast in view). Treating it as a CC with a "no mass" label could create a false negative, so it is not used. The meaning of `FB` comes from the clinical description, not from the dataset documentation.

### Breast bounding box

The breast covers about a third of each image (threshold on intensity above 100; background is close to 0). Its bounding box covers a median of 44% of the image (range 15% to 75%) and is about 3150 x 1470 px at the median. Cropping to the box before resizing keeps more resolution for the same cost. This is a rough threshold estimate over all 410 images; it needs a visual check before use.

## Annotations

The XML files are Apple property lists (`plistlib` reads them). Each ROI has a name, a type and polygon points in pixel coordinates `(x, y)` = `(column, row)`. Overlays on a sample of images confirm there is no flip or offset.

- Images without an XML file are exactly the 67 BI-RADS 1 images (no lesions).
- 7316 ROIs in 343 images. Most are calcifications.

The names are normalized to a closed set of lesion types (misspellings merged, see `src/mammo_seg/data/annotations.py`). Counts by lesion type:

| Lesion type | ROIs | Images | Shape |
|---|---|---|---|
| calcification | 7143 | 313 | 4339 single points, 2800 small polygons (median 9 points, about 88 px area), a few larger shapes |
| mass | 116 | 107 | polygon (median 116 points) |
| cluster | 27 | 21 | 200-point ellipse-like region |
| spiculated_region | 16 | 16 | 200-point region |
| asymmetry | 6 | 6 | 200-point region or polygon |
| distortion | 3 | 3 | 200-point region |
| unknown | 5 | 4 | no usable name: `Point 1`, `Point 3`, `Unnamed` (2) and one empty name |

Raw spellings that are merged: `Calcifications`, `Spiculated region`, `Espiculated Region` and `Assymetry`. A name that is not in the mapping raises an error when the manifest is built.

### Why not calcifications (for now)

There are more of them (7143 annotations in 313 images), but 4339 are single points with no extent and the polygons are tiny (about 10 px across at native size). They cannot be downscaled, masks would be dilated points (weak labels), and Dice is not a meaningful metric for them. They need full-resolution patches and detection-style metrics (lesion-level precision and recall, FROC). Cluster regions (27 in 21 images) could join that task.

## Masses

- 116 masses in **107 images from 50 patients**. 99 images have one mass, 7 have two and 1 has three. Views are even: 54 CC, 53 MLO.
- Size (equivalent diameter at native resolution): min 62 px, 5th percentile 103, median 234, 95th percentile 706, max 978. Area is 0.03% to 5.8% of the image (median 0.4%).
- Class imbalance: masses are **0.27% of all pixels** over the whole dataset (1.0% over images that contain a mass).
- BI-RADS does not select masses: they occur from BI-RADS 2 upward. Positive images by BI-RADS: 2: 22, 3: 13, 4a: 7, 4b: 2, 4c: 12, 5: 43, 6: 8.
- Three polygons extend past the image border by a few pixels, so they must be clipped when rasterized.

### Size after downscaling

Masses whose equivalent diameter falls below a size after downscaling (116 masses in total):

| Downscale | Min | Median | Under 16 px | Under 32 px |
|---|---|---|---|---|
| /1 | 62 | 234 | 0 | 0 |
| /2 | 31 | 117 | 0 | 1 |
| /3 | 21 | 78 | 0 | 4 |
| /4 | 15 | 59 | 1 | 18 |
| /8 | 8 | 29 | 18 | 64 |
| /16 | 4 | 15 | 64 | 102 |

Up to /4 the masses stay clearly visible; at /8 and beyond a sixth to over half become very small. Cropping to the breast bounding box first (see above) allows the same output size with a smaller factor: a crop resized to a longer side of about 1024 px is roughly /3 and keeps the smallest mass at about 20 px.

## Positive and negative images, patients

An image is **positive** if it has at least one mass, otherwise **negative**.

| | Images |
|---|---|
| Positive | 107 |
| Negative | 303 |
| of which normal, no annotation (BI-RADS 1) | 67 |
| of which annotated, calcifications or other lesions only | 236 |

Positives by view: 54 CC and 53 MLO; by side: 55 left and 52 right; by size: 57 at 3328 x 2560 and 50 at 4084 x 3328.

Patients (108):

| Images per patient | No mass image | Mixed (some images positive) | All images positive |
|---|---|---|---|
| 2 | 16 | 1 | 8 |
| 3 | 1 | 0 | 0 |
| 4 | 36 | 35 | 4 |
| 7 | 0 | 1 | 0 |
| 8 | 5 | 0 | 0 |
| 10 | 0 | 1 | 0 |
| **Total** | **58** | **38** | **12** |

- 50 patients have at least one positive image (38 mixed + 12 all positive) and 58 have none. In mixed patients the mass is in one breast and the other breast's images are negative.
- Positive images per positive patient: 1 for 2 patients, 2 for 43, 3 for 1, 4 for 4.
- The same mass is usually visible in both views: of 52 breasts with two views and a mass, 51 show it in both CC and MLO. The two views are near-duplicates.

## Implications for the project

1. **Scope: mass segmentation.** Masses are the only lesion with a true region annotation in enough images (see above for calcifications).
2. **Very small training set.** 107 positive images from 50 patients. Use cross-validation grouped by patient, report variability across folds, and expect noisy metrics. A pretrained encoder and strong augmentation are likely needed.
3. **Splitting.**
   - Group by patient id, never by image or breast: both views of a mass, both breasts, and the second exam of 9 patients would otherwise leak.
   - Stratify at patient level on "has any mass image" (50 positive, 58 negative patients). With 5 folds that is about 10 positive patients and 21 positive images per fold. `StratifiedGroupKFold` from scikit-learn does this.
   - Optional extra balance: image size and BI-RADS.
   - There is no official split. A fixed held-out test set of about 20% of patients would hold only about 10 positive patients, so metrics from it are very noisy. Report cross-validation results and use a test set only as a sanity check.
4. **Negatives matter.** Of the 410 images, 303 have no mass. Decide whether to train on positives only or on all images, and report lesion-positive and all-image metrics separately. Also report the 67 normal and the 236 other-lesion negatives separately, since they behave differently.
5. **Masks must be derived.** Rasterize the polygons (clip to the image, merge multiple masses per image) and validate them. Keep the derived masks under `data/processed/`.
6. **Preprocessing choices to make:** one orientation for both breasts, intensity scaling from the 12-bit range, cropping to the breast, and the offline downscale (a factor of /2 to /4 of the native image, or a crop resized to about 1024 px).
7. **Grouping key:** patient id from the filename, never the file id.

## Open questions

- Is the Kaggle copy identical to the original release (108 patients here against 115 cases in the publication)?
- Pixel spacing and bit depth: confirm against `inbreast.pdf`.
- Does the breast bounding box (threshold 100) hold up on a visual check of all images?
