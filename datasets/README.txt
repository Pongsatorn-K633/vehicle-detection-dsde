Prepared training data (only needed for training; testing with the included weights does not use it)

If you received this project as a ZIP, the prepared data is already here: skip data preparation.
It is not in the GitHub repo (about 2 GB); get it from the project's Google Drive folder and unzip it in the
project root (each zip extracts into datasets/):

  Folder               Google Drive zip                    Used by
  rfdetr_full/         shared_detector_data_v1_v2.zip      RF-DETR training (COCO format), versions 1 and 2
  yolo_full/           shared_detector_data_v1_v2.zip      YOLO26m training (YOLO format), versions 1 and 2
  external/            v2_external_prepared.zip            version 2 classifier: 11,838 images, 16,828 boxes

rfdetr_full/ and yolo_full/ hold the 2,991 Kaggle train images of original-data/ (all 15 cameras; rare classes
copied up to 4 times). To rebuild them instead:
  python src/prepare_data.py --full

external/ comes from nine Roboflow datasets (sources and class mapping: EXTERNAL_DATA.md). To rebuild it, download
them into external-data/ and run:
  python src/prepare_external.py

Both commands delete and rewrite their output folder.
