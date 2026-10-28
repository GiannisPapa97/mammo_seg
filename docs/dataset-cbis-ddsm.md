# Dataset: CBIS-DDSM (Kaggle copy)

Findings on the raw data in `data/raw/cbis-ddsm/`, from exploration scripts and the manifest built by `mammo_seg.data.cbis_ddsm` (see [data-pipeline.md](data-pipeline.md)). Only aggregate figures are given. Source: [awsaf49/cbis-ddsm-breast-cancer-image-dataset](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) on Kaggle, which repackages the CBIS-DDSM collection of The Cancer Imaging Archive (TCIA) as JPEG images. CBIS-DDSM (Curated Breast Imaging Subset of DDSM) is a curated part of the Digital Database for Screening Mammography: scanned film mammograms.

**Scope:** the project uses the **masses** of CBIS-DDSM, to pretrain before INbreast. Only the manifest is built so far; the masks, the preprocessing and the splits do not read it yet.

## License and use

Checked on 2026-10-28 (not legal advice):

- **The original (TCIA):** the [collection page](https://www.cancerimagingarchive.net/collection/cbis-ddsm/) gives [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) for the images and the CSV files: use, also commercial, with attribution. Users must follow the [TCIA Data Usage Policies and Restrictions](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/): cite the dataset with its DOI whenever the data are used, name it and where it was obtained in presentations and publications, and never use it to identify or contact the patients.
- **The Kaggle copy** used here lists CC BY-SA 3.0 in its metadata, while its description says the CC BY 3.0 of TCIA applies. Both require attribution; ShareAlike would also require that derived data shared publicly (converted images, masks, manifests) be shared under the same license. Nothing derived is shared, so the difference does not matter so far; if that changes, take the data from TCIA, under CC BY 3.0 alone.

Citations to give in any publication or report of results:

- Data: Sawyer-Lee, R., Gimenez, F., Hoogi, A., & Rubin, D. (2016). *Curated Breast Imaging Subset of Digital Database for Screening Mammography (CBIS-DDSM)* [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY
- Publication: Lee, R. S., Gimenez, F., Hoogi, A., Miyake, K. K., Gorovoy, M., & Rubin, D. L. (2017). A curated mammography data set for use in computer-aided detection and diagnosis research. *Scientific Data*, 4, 170177. https://doi.org/10.1038/sdata.2017.177
- TCIA: Clark, K., et al. (2013). The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository. *Journal of Digital Imaging*, 26(6), 1045–1057. https://doi.org/10.1007/s10278-013-9622-7

As for INbreast, no samples go in the repository, tests or docs, and `data/` stays git-ignored: the license would allow sharing with attribution, but the repository does not need it, and the INbreast data next to it cannot be shared.

## Contents

| Folder / file | Content |
|---|---|
| `csv/mass_case_description_{train,test}_set.csv` | One row per mass: patient, breast, view, density, abnormality id, shape, margins, BI-RADS assessment, pathology, subtlety, and the DICOM paths of the full image, the crop and the mask |
| `csv/calc_case_description_{train,test}_set.csv` | The same for calcifications (not read) |
| `csv/dicom_info.csv` | The DICOM header of every file, with its JPEG path (not read) |
| `csv/meta.csv` | One row per DICOM series (not read) |
| `jpeg/<series>/` | 10,237 8-bit grey JPEGs, one folder per DICOM series |

## Linking the CSVs to the JPEGs

- The CSV paths have the form `<folder>/<study>/<series>/<file>.dcm`. The DICOM files are not in the Kaggle copy; the JPEGs of a series are in `jpeg/<series>/`.
- The folder of a full image names it: `Mass-<Training|Test>_<patient>_<LEFT|RIGHT>_<CC|MLO>`, which agrees with the columns of the row on every row. The folder of a mass adds `_<abnormality id>`.
- The series of a full image holds one JPEG (all 1,592). The series of a mass holds its mask and its crop; the mask is the file with the size of the full image.
- The mask paths in the CSVs end with a newline, so the paths are stripped.

## Masses

- The mass CSVs list **1,696 masses on 1,592 images of 892 patients**: 1,318 masses on 1,231 images of 691 patients in the official train set, 378 on 361 images of 201 patients in the test set. **No patient is in both sets.**
- Views: CC and MLO only. Density (`breast_density`): 1 to 4, the same on every row of an image. Assessment 0 to 5, subtlety 0 to 5, pathology `MALIGNANT`, `BENIGN` or `BENIGN_WITHOUT_CALLBACK`. Shape and margins are combinations such as `IRREGULAR-ARCHITECTURAL_DISTORTION`; `N/A` means no value.
- **78 masses, on 78 images, have no mask of the image's size**: in 74 the series holds only the crop, in 4 its two files both have another size. These images are left out of the manifest (`dropped`; 65 train, 13 test), because a mass without a mask would be trained as background.
- Masks are nearly binary: no pixel of any mask is strictly between 30 and 225, and at most 0.16% of the pixels are neither 0 nor 255 (JPEG compression at the edges). A pixel above 127 is mass.
- A mass covers a median 0.36% of its image (from 0.017% to 5.1%; 0.13% and 1.0% at the 10th and 90th percentiles). The images are large: height 3,920 to 6,931 pixels (median 5,266), width 1,786 to 5,431 (median 3,008).

## The manifest

| | Images | Masses | Patients |
|---|---|---|---|
| Train | 1,166 | 1,253 | 657 |
| Test | 348 | 365 | 196 |
| All | 1,514 | 1,618 | 853 |

Every image has a mass (1,443 have one, 71 two to six). 768 masses are malignant, 709 benign and 141 benign without callback. The shape is `N/A` for 4 masses and the margins for 60.

## What it does not have

- **No pixel size.** The JPEGs keep none, and the films of DDSM were scanned by several scanners at different resolutions. The manifest has `null`, so no step computes areas in mm² until the pixel size is settled (for the preprocessing).
- **No acquisition date.** The dates of `dicom_info.csv` (2016 and 2017) are those of the conversion.
- **No BI-RADS of the whole image**, only the assessment of each mass.
- **No images without a mass.** The images with calcifications only are in the calcification CSVs, which are not read; the calcifications on the mass images are not annotated in the manifest.

## Differences from INbreast

| | INbreast | CBIS-DDSM (masses) |
|---|---|---|
| Images | 410 full-field digital, 16-bit DICOM | 1,514 scanned films, 8-bit JPEG |
| Patients | 108 | 853 |
| Masses | 116, on 107 images | 1,618, on 1,514 images |
| Outlines | polygons | mask images |
| Pixel size | 0.07 mm | unknown |
| Images without a mass | 303 | none (in this manifest) |
| Official split | none | train and test |
