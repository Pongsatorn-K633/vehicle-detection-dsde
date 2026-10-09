# External data

As of 2026-10-05. The TA confirmed that external data is allowed. The test images are newly labelled BMA
CCTV frames (25–31 Aug 2026), and the private test set comes from different cameras "of the same
characteristics", so external data is judged by how close it is to that CCTV look.

Nine Roboflow datasets (YOLO export) were downloaded into `external-data/` (not in git, 3.8 GB of zips).
**The 0.703 submission uses none of it; the 0.707 submission uses it in the classifier only.** Any use is first tested on the Round 1 validation split
(baseline val mAP50 **0.688**) and kept only if it beats that.

## 1. Datasets

"Unique images" counts each source image once. Four sets contain Roboflow's pre-made augmented copies
(3 versions of each image), which are removed. Box size = sqrt(box area / image area), median.

| Dataset (folder) | Roboflow source | Unique images | Look | Boxes per image | Box size | Fit for the test | Use |
|---|---|---|---|---|---|---|---|
| `vehicle_car` | training-lfouo/vehicle_car | 1,919 | Thai CCTV intersections, some close-ups | 2.2 (partial) | 0.14 | best | detector + classifier |
| `traffic_count` | project-1-mfrmq/traffic-count-d7xe6 | 1,219 | CCTV, highway (not Bangkok) | 2.6, many vehicles unlabelled | 0.10 | good | detector (labels filled) + classifier |
| `vehicle_detection_v22` | capstonedesign-rotpc/vehicle-detection-yg4le-c24ru | 2,444 | close-up photos, ground level | 1.4 | 0.48 | poor | classifier only |
| `vehicle_detection_v1` | thaidetec/vehicle-detection-yg4le | 1,830 | close-up photos | 1.2 | 0.67 | poor | classifier only |
| `classification_of_cars` | thiraphat/classification-of-cars | 7,173 | stock photos (iStock, Alamy), close-up | 1.1 | 0.76 | poor, labels unreliable | classifier only |
| `songthaew` | (songthaew v2) | 50 | street photos, mostly red Chiang Mai type | 1.4 | 0.38 | poor | classifier only |
| `tuktuk_detection` | vehicle-detection-8ljxf/tuk-tuk-detection | 300 | close-up, heavy HDR filter | 1.5 | 0.38 | poor | not used (Tuktuk already 0.87) |
| `test_mlejl` | test-coqzq/test-mlejl | 155 | close-up, mostly rickshaws and sidecars | 1.6 | 0.56 | poor | a few Tuktuk / Songthaew crops |
| `bangkok_bus` | saint70239/bangkok-bus-dataset-type2 | 1,099 | phone photos of buses | **0 (no labels)** | – | – | **not usable** |
| *Ours (`train.csv`)* | | *2,991* | *BMA CCTV, 352 × 288, high and far* | *9.5* | *0.05* | | |

Checks:
- **No leakage:** 0 external images match any of our train or test images (perceptual hash).
- **Little overlap between sets:** 18 images are shared by v1 and v22; a few others appear in 2 sets. Duplicates
  are kept once.

Why most sets are classifier-only: the detector learns what a whole CCTV frame looks like (many small vehicles,
seen from above). Close-up photos of one large vehicle teach it a different picture. The classifier sees one
cropped vehicle at a time, so close-ups can still teach it what tells a pickup, a songthaew or a truck apart.

## 2. Class mapping to our 8 classes

Our definitions (competition PDF):

| ID | Class | Definition |
|---|---|---|
| 0 | Car | passenger car, 4-door, 5-door, standard SUV |
| 1 | Motorcycle | one box for the rider and passengers |
| 2 | Bus | bus, coach, large passenger vehicle |
| 3 | Truck | semi-truck, trailer, dump truck, container, closed box truck (รถตู้ทึบ), 6+ wheels |
| 4 | Tuktuk | three-wheeled motorized passenger vehicle |
| 5 | Van | passenger van |
| 6 | Pickup | standard pickup truck for personal use |
| 7 | Songthaew | modified pickup or small truck with two rows of passenger seats in the rear |

**Convention taken from our training labels:** a pickup with a closed box or canopy on the back is labelled
**Truck** in `train.csv`, so external box-pickups also go to Truck.

### Kept

| Our class | External class names |
|---|---|
| Car | `car`, `SUV`, `sedan`, `hatchback`, `jeep`, `taxi`, `supercar`, `suv`, `PPV`, `Car` (test_mlejl), `Car1` (traffic_count, all SUVs) |
| Motorcycle | `motorbike`, `motorcycle`, `Motorcycle`, `MC` |
| Bus | `bus`, `Bus_L` |
| Truck | `10 wheel large truck`, `6 wheel medium truck`, `semi trailer truck`, `tow truck` (all big trucks), `Six-wheeled vehicle bus` (6-wheel trucks despite the name), `truck` (vehicle_car, v1, v22), `Trailer`, `Truck_L`, `Truck_m`; box-pickups `Two-section pickup truck`, `Solid box pick-up` |
| Tuktuk | `tuktuk`, `3WTuktuk` |
| Van | `van`, `Van`, `MPV` |
| Pickup | `pickup`, `pick-up`; `truck` in **classification_of_cars** (these are all pickups) |
| Songthaew | `songthaew`, `Songthaew` (all sets, including traffic_count's 27 rural-style boxes: pickups with a roof frame and passengers in the rear, kept after review) |

### Dropped

| External class | Why |
|---|---|
| `bicycle` | not one of our classes |
| `Samlor`, `3-wheel-bike` | pedal rickshaws, not motorized tuktuks |
| `saleng`, `Sidecar`, `Puangkang` | motorcycle sidecar carts, not tuktuks |
| `e_tan` | farm vehicle (อีแต๋น) |
| `ambulance` | mix of vans and pickups |
| `boxtruck` (v1, v22) | dropped by decision; mixed box-pickups and canopies |
| `4 wheel small truck`, `mini_truck` | kei trucks (Suzuki Carry type), no matching class |
| `4WTuktuk` | small 4-wheel taxi, no matching class |
| `3 wheel motorbike` | mix of tuktuks and cargo trikes; first filtered to tuktuk-looking crops, then dropped entirely (val 0.710 with them, 0.713 without) |
| all of `bangkok_bus` | no labels |

### Re-labelled by our model

| External class | Problem | Handling |
|---|---|---|
| `Car0` (traffic_count) | mixes cars and pickups | box kept, class predicted by our 8-class classifier |
| `Truck_s` (traffic_count) | mixes small trucks, box-pickups and vans | same |
| `Bus_s` (traffic_count) | mixes minibuses and vans | same |

Re-labelled boxes are used only for the detector (so the vehicle is not treated as background), never to
train the classifier, which would only learn its own guesses back.

## 3. How the data is prepared

1. One copy per source image (Roboflow `.rf.<hash>` copies removed); cross-set duplicates kept once.
2. Polygon labels converted to their bounding box.
3. Class names mapped with the tables above; dropped classes removed.
4. `traffic_count` mixed classes re-labelled by our classifier, limited to the classes each label can be
   (Car0: Car/Pickup/Truck, Truck_s: Truck/Pickup/Van, Bus_s: Bus/Van). Unrestricted, it called dark SUVs "Bus".
5. For the detector (CCTV-like sets only): vehicles the external labels missed are filled in with our Round 1
   detectors, and images are shrunk to our scale (objects about 5% of the image).

## 4. Expected gain per class

Unique boxes after mapping (approximate, before the tuktuk filter and re-labelling).

| Class | Our train boxes | Val AP50 | External boxes | of which CCTV-like | Value |
|---|---|---|---|---|---|
| Pickup | 497 | 0.50 | ~2,000 | ~790 | highest: biggest loss, real CCTV pickups available |
| Songthaew | 117 | 0.17 | ~710 | ~220 | medium: most external ones are the red Chiang Mai type, ours are the larger Bangkok type |
| Truck | 1,356 | 0.69 | ~2,700 | ~210 | medium |
| Van | 354 | 0.75 | ~1,240 | ~500 | medium |
| Bus | 482 | 0.76 | ~640 | ~140 | low to medium |
| Tuktuk | 265 | 0.87 | ~850 | 0 | low |
| Car, Motorcycle | 17,762 / 6,563 | 0.94 / 0.86 | ~9,000 | – | none needed |

## 5. Experiments

All on the Round 1 validation split (3 held-out cameras), same detectors (RF-DETR Large 12 epochs + YOLO26m
25 epochs, last epochs), only the classifier changes. `datasets/external/external.csv`: 12,308 images and
18,867 boxes; 16,828 used for the classifier (re-labelled boxes excluded).

| Classifier | External crops | Classifier balanced acc | Best alpha | Val mAP50 |
|---|---|---|---|---|
| 8-class, no external (current) | 0 | 0.671 | 0.4 | 0.688 |
| A2: + CCTV-like sets only | 5,381 | 0.687 | 0.4 | 0.697 |
| A1: + all usable sets, incl. filtered `3 wheel motorbike` | 16,926 | 0.694 | 0.6 | 0.710 |
| **A1b: + all usable sets, `3 wheel motorbike` dropped** | **16,828** | **0.694** | **0.6** | **0.713** |

Alpha sweeps: A1 0.2 → 0.690, 0.4 → 0.703, 0.6 → 0.710, 0.7 → 0.711, 0.8 → 0.705; A1b 0.4 → 0.705,
0.5 → 0.711, 0.6 → 0.713, 0.7 → 0.713. Validation preferred 0.6, but on Kaggle 0.6 scored lower than 0.4
(table below), so the final setting is **alpha 0.4**, the same as before external data: the classifier is less
reliable on the unseen test cameras than on our validation cameras, and should not be trusted more.

Kaggle public scores (same Round 2 detectors, only the classifier and alpha change):

| File | Classifier | Alpha | Val mAP50 | Public |
|---|---|---|---|---|
| `wbf_final_c8big_a0.4.csv` | no external | 0.4 | 0.688 | 0.7035 |
| **`wbf_final_ext_a0.4.csv`** | **external (A1b)** | **0.4** | **0.705** | **0.7070** |
| `wbf_final_ext_a0.6.csv` | external (A1b) | 0.6 | 0.713 | 0.7019 |

Per class on validation, A1b at alpha 0.6 against the previous classifier at 0.4:

| Class | Current | A1b | Change |
|---|---|---|---|
| Car | 0.940 | 0.940 | 0.000 |
| Motorcycle | 0.861 | 0.860 | 0.000 |
| Bus | 0.777 | 0.789 | +0.012 |
| Truck | 0.720 | 0.728 | +0.008 |
| Tuktuk | 0.849 | 0.855 | +0.006 |
| Van | 0.697 | 0.786 | +0.089 |
| Pickup | 0.499 | 0.534 | +0.035 |
| Songthaew | 0.159 | 0.214 | +0.055 |

Caution: Van has 31 validation boxes and Songthaew 24, so their changes are noisy. The gain is spread over
every rare class, and the close-up sets (A1) beat the CCTV-only sets (A2), so for the classifier the different
camera angle is not a problem.

### Experiment B: detectors with external CCTV frames

Detector data: `src/prepare_external_det.py`. `vehicle_car` and `traffic_count` only; 713 close-up images
skipped (largest box above 35% of the image side); 4 images resized to 176 × 144 and tiled into one 352 × 288
frame; missing vehicles filled in by the Round 1 detectors (fused score ≥ 0.35, no overlap with a labelled box).
Result: 593 frames, 9,464 boxes (3,108 filled in), median box 4.6% of the frame (ours 5.0%). Added once each to
the Round 1 train split (`prepare_data.py --external`); validation unchanged. Same training settings.

| Pipeline (val mAP50) | Detector only, no TTA | + flip TTA, WBF 2:1 | + external classifier, alpha 0.4 | alpha 0.6 |
|---|---|---|---|---|
| RF-DETR old + YOLO old | 0.625 / 0.566 | 0.676 | **0.705** | **0.713** |
| RF-DETR **new** + YOLO old | 0.634 / – | 0.686 | **0.706** | 0.711 |
| RF-DETR old + YOLO **new** | – / 0.531 | 0.661 | 0.694 | 0.703 |
| RF-DETR **new** + YOLO **new** | 0.634 / 0.531 | 0.672 | 0.699 | 0.705 |

- **RF-DETR** gains on its own (0.625 → 0.634, mostly Tuktuk and Songthaew), but the external classifier already
  fixes those classes, so after the classifier the gain is gone (alpha 0.4: 0.705 → 0.706; alpha 0.6:
  0.713 → 0.711; both within noise).
- **YOLO** gets worse (0.566 → 0.531): Truck −0.08, Van −0.08. The tiled frames (stretched images, visible tile
  seams, filled-in labels) likely hurt it more than the transformer.
- **Decision: not kept.** The final models stay the Round 2 detectors trained on our data only.

| Experiment | Status |
|---|---|
| A. Classifier with external crops (A1b) | **kept** at alpha 0.4: val 0.688 → 0.705, public 0.7035 → **0.7070**; final file `submissions/v2_external_data/wbf_final_ext_a0.4.csv` |
| B. Detectors with external CCTV frames | **not kept**: at alpha 0.4, 0.706 vs 0.705 (tie); new YOLO always lower |
