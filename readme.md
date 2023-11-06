---
contributors:
  - Hugo Storm
  - Thomas Heckelei
  - Kathy Baylis
  - Klaus Mittenzwei
---

## Overview

This repository contains the replication package for the paper "Identifying farmers’ response to changes in marginal and average subsidies using deep-learning" by Hugo Storm, Thomas Heckelei, Kathy Baylis, and Klaus Mittenzwei. 

The main data file including the farm level activity levels is confidential and cannot be shared. This readme provides information about how to obtain access to the data. The code is provided to replicate the results in the paper, a dummy dataset is provided with the same structure of the original file. WIth this file the code can be run.


## Data Availability and Provenance Statements

### Statement about Rights

- [x] I certify that the author(s) of the manuscript have legitimate access to and permission to use the data used in this manuscript. 
- [ ] I certify that the author(s) of the manuscript have documented permission to redistribute/publish the data contained within this replication package. Appropriate permission are documented in the [LICENSE.txt](LICENSE.txt) file.

### License for Code and Data
The provided code and data are licensed under a Creative Commons/CC-BY-NC license. See LICENSE.txt for details.

### Summary of Availability

- [ ] All data **are** publicly available.
- [x] Some data **cannot be made** publicly available.
- [ ] **No data can be made** publicly available.

### Details on each Data Source

| Data.Name  | Data.Files | Location | Provided | 
| -- | -- | -- | -- | 
| "Farm Activities" | `balancedPanel_dummy.csv` | `data/raw/` | FALSE | 
| "Subsidy Rates" |  `satser.csv`; `bunn.csv`; `maks.csv`; `trin.csv` |  `data/raw/` | TRUE | 
| "Pasture Payments" |  `grovfac.csv` |  `data/raw/` | TRUE | 
| "Prices" |  `importEAAPrices.csv` |  `data/raw/` | TRUE | 

#### Farm Activities
Data file: `balancedPanel_dummy.csv`

The data of farm level activities used for this project are confidential, but may be obtained with Data Use Agreements with the Norwegian Agriculture Agency [NAA] (www.landbruksdirektoratet.no, in Norwegian only). Researchers interested in access to the data may contact co-author Klaus Mittenzwei to establish communication with contact persons at NAA. Once a Data Use Agreement is obtained for the variables listed under “Codebook data/raw/balancedPanel.csv“ the authors can provide a preprocess data file that allows to replicate the analysis. For testing the code a dummy dataset is provided. This dummy data set has the same structure as the original data but contains randomly generated values. 

#### Subsidy Rates
Data files: `satser.csv`; `bunn.csv`; `maks.csv`; `trin.csv` 

This information about subsidy rates are derived by the authors from the following sources:

NIBIO (1999-2016) Referansebruksberegninger, ISSN 0804-4201, available online at https://nibio.brage.unit.no/nibio-xmlui/discover?rpp=10&etal=0&query=REFERANSEBRUKSBEREGNINGER&group_by=none&page=1

The derived dataset are made available under a Creative Commons Non-commercial license. 

#### Pasture Payments

Data file: `grovfac.csv`

This information about Pasture Payments are derived by the authors from the following sources:

Landbruksdirektoratet, Avgrensning av tilskuddsberettiget grovfôrareal og innmarksbeiteareal, available online at https://www.landbruksdirektoratet.no/nb/jordbruk/ordninger-for-jordbruk/produksjonstilskudd-og-avlosertilskudd-i-jordbruket/produksjonstilskudd-og-avlosertilskudd--beregningsveiledning/9.areal-og-kulturlandskapstilskudd#avgrensning_av_tilskuddsberettiget_grovf%C3%B4rareal_og_innmarksbeiteareal  

The derived dataset are made available under a Creative Commons Non-commercial license. 

#### Prices 
Data file: `importEAAPrices.csv` 

Price indicies are derived by the authors from the following source:

NIBIO, Totalkalkylen - statistikk, available online at https://www.nibio.no/tjenester/totalkalkylen-statistikk#groups

The derived dataset are made available under a Creative Commons Non-commercial license. 

### Codebook for different data sources

#### Codebook  `data/raw/satser.csv`:
| Code| Description |
|-----------|--------|
| year | year|
| cat_sub | category of subsidy |
| cat_act | activity |
| step | step of subsidy |
| zone | zone of subsidy |
| rate | rate per head or area |

#### Codebook `data/raw/trin.csv`
| Code| Description |
|-----------|--------|
| year | year|
| cat_sub | category of subsidy |
| cat_act | activity |
| step | step of subsidy |
| cut | Upper limit of step |

#### Codebook `data/raw/maks.csv`
| Code| Description |
|-----------|--------|
| year | year|
| cat_sub | category of subsidy |
| rate | Upper limit of payments for TVELF and TPROD payments |

#### Codebook `data/raw/bunn.csv`
| Code| Description |
|-----------|--------|
| year | year|
| cat_sub | category of subsidy |
| rate | Overall discount deducted from the TAKTL payments (Bunnfradrag) |

#### Codebook `data/raw/grovfac.csv`
| Code| Description |
|-----------|--------|
| year | year|
| cat_sub | category of subsidy |
| cat_act | activity |
| zone | zone of subsidy |
| rate | rate per head or area |

#### Codebook `data/raw/balancedPanel.csv`
| Code| Description |
|-----------|--------|
| KGB | farm identifier (encoded) | 
| year | year| 
| knr | region number | 
| age | age of farmer in years| 
| x1 | Apples (kg)| 
| x2 | Pears (kg)| 
| x3 | Plums (kg)| 
| x4 | Cherries ( Prunus avium) sweet (kg)| 
| x5 | Cherries (Prunus cerasus) sour (kg)| 
| x6 | Apples/ Pears (kg)| 
| x11 | Strawberries (kg)| 
| x12 | Raspberries (kg)| 
| x13 | Blackcurrant (kg)| 
| x14 | Currants (kg)| 
| x16 | Blueberries (kg)| 
| x21 | Gooseberry (kg)| 
| x31 | Tomato (kg)| 
| x32 | Cucumber (kg)| 
| x33 | Lettuce (head) | 
| x60 | Potatoes (kg) | 
| x111 | adult horses | 
| x112 | other horses | 
| x113 | young horses under 2 yrs | 
| x114 | horses for breeding | 
| x115 | Horses under 3 years | 
| x116 | Horses over 3 years | 
| x120 | Dairy cows |
| x121 | Suckler cows for special meat production |
| x133 | Sheep or lamb which is kept outside most of the year| 
| x136 | Lamb (i.e., sheep under 1 year) that is kept inside during vinter | 
| x140 | Female goat over 1 year /Milkgoat |
| x142 | Suckler goats for special meat production | 
| x155 | Sows for breeding with minimum one litter |
| x157 | Slaughter pigs| 
| x160 | Laying hens at counting date / over 20 weeks |
| x180 | rabbits | 
| x181 | Deer farming, own manufactured grassland | 
| x183 | Ostrich | 
| x186 | Poultry sold as living animal | 
| x192 | Donkey | 
| x193 | Horses in pens. In the grazing season | 
| x196 | lama | 
| x197 | alpaca | 
| x230 | Potatoes |
| x235 | Meadow seed and other seed production | 
| x236 | Peas, beans and other legumes to… (mature?) | 
| x237 | Oil seeds | 
| x238 | Rye and rye wheat | 
| x239 | Grain to crush | 
| x240 | Wheat (Spring Wheat) | 
| x242 | Barley | 
| x243 | Oats | 
| x245 | Peas and beans to konserves | 
| x246 | Meadow seed, seeds, peas and beans ripening, Flaxseed | 
| x247 | Autumn Wheat | 
| x248 | Oil seeds seeded in autumn | 
| x249 | rye wheat | 
| x410 | Dairy Cows on outlying fields | 
| x411 | Dairy Cows on meadow | 
| x420 | Other livestock on outlying fields /Young cattle | 
| x422 | Other livestock on meadow 12/16 weeks, Young cattle | 
| x440 | Goat on outlying fields | 
| x445 | Goat on meadow 12/16 weeks| 
| x450 | Horses on rangeland | 
| x471 | lam slaughter 2011, quality 0 or better | 
| x473 | lam slaughter 2011, quality 0- | 
| x474 | lam slaughter 2011, quality P+ or worse | 
| x475 | KJE slaughter 2011, weight over 3,5 kg | 
| x487 | Sheep on meadow 1 year or older on meadow 12/16 weeks | 
| x488 | Lamb and sheep under 1 year old on meadow 12/16 weeks| 
| x521 | Sale of feed, High (kg) | 
| x522 | Sale of feed,  silage (kg) | 
| x523 | haylage (kg) | 
| x832 | slaugher pigs, pigs intendend for breeding | 
| x841 | laying hens | 
| CERE | Cereals (Codes: 235, 236, 237,238,239,240,242,243,245,246,247,248,249) |
| GEIT | male goat (Codes: 141,143,144) |
| GRON | Vegetables outside (Codes: 260,263,264) |
| SAU | Sheep (Codes: 130, 137 and 149) |
| STOR | Other Cattle (Codes: 119,122,123,124,125,126,127,128,129) |
| BAER | Berries (Codes: 280, 281, 282, 283)| 
| FODD | Fodder (210,211,212) | 
| FRUK | Fruits (Codes: 271, 272, 273, 274)| 
| HEST | Horses (Codes: 111,112, 113,114,115,116) | 
| USAU | Sheep on outlying fields (Codes: 430,431,432,437,438)| 
| VSAU | Sheep kept inside during winter (Codes: 134,135,145,146) | 
| x210 | Fodder on arable land | 
| x211 | Fodder (pasture) on arable non-fenced land | 
| x212 | Fodder on non-arable fenced land | 
| x213 | Other fodder | 
| zoneTAKTL | zone relevant for TAKTL subsidies | 
| zoneTDISE | zone relevant for TDISE subsidies | 
| zoneTDISG | zone relevant for TDISG subsidies | 
| zoneTDISK | zone relevant for TDISK subsidies | 
| zoneTDISM | zone relevant for TDISM subsidies | 
| zoneTDMLK | zone relevant for TDMLK subsidies | 
| zoneTPROD | zone relevant for TPROD subsidies | 
| birthyear | year of birth farmer| 


## Computational requirements

To replicate the software environment, use the provided Dockerfile. You can either pull the image from Docker Hub or build it yourself.

To use the provided docker image, you need to have docker installed on your machine. See https://docs.docker.com/get-docker/ for details.

The application is build for a Linux environment. It has been tested on a machine with the following hardware:
- CPU: Intel(R) Xeon(R) W-2155 CPU @ 3.30GHz
- GPU: 3x NVIDIA RTX A5000
- RAM: 128 GB

#### Pull docker image from Docker Hub
Pul and run the container: `docker run --gpus all -it hstorm/nn_norway_ajae:1.0`

#### Rebuild docker image locally
Pull the repository from GitHub: `git clone https://github.com/hstorm/nn_norway_ajae.git` 
Build docker image: `docker build -t nn_norway_ajae:1.0 .`

#### Run docker image
To Start container: `docker run --gpus all -it nn_norway_ajae:1.0`

Once the container is running, you can attach a shell and run the code as described below.

### Software Requirements

Software requirements are listed in the `Dockerfile`. Python requirements are listed in `requirements_docker.txt`.
When using the provided docker image from Docker Hub, all software requirements are installed.


### Controlled Randomness

Controlling of randomness is not fully possible when training on a GPU, one reason is that controlling randomness would required a exactly equal hardware setup.

### Memory and Runtime Requirements

Training of the model can best be performed using a single GPU (e.g. NVIDIA RTX A5000) but training on a CPU is also possible. 

## Description of programs/code

Organization of the code largely follows the Cookiecutter Data Science template (https://drivendata.github.io/cookiecutter-data-science/)

Most important parts are:
- Programs in `src/features` are used to prepare the features. 
- Programs in `src/models` are used to train models and perform the post-model analysis and to prepare the results. 

### License for Code

The code is licensed under a MIT license. See [LICENSE.txt](LICENSE.txt) for details.

## Instructions to Replicators


#### Build dataset 
On a first run the dataset needs to be build. This can be done by running the following programs in the following order:

Open two shells (at the same time) and run
1) `scl enable rh-python36 -- python luigid`
2) `scl enable rh-python36 -- python src/features/luigi_features.py`

For details refer to the in-code documentation in: `src/features/luigi_features.py` and `src/features/calc_dpay.py`


#### Train a model
To train a model run the following program:
`scl enable rh-python36 -- python src/models/nn_model.py` 


### Details

To reproduce the results in the paper use the trained model to create the figures (see commands below). The ID of the model used for the paper is `15283bd4-51d2-11ec-8639-0242ac110003`, stored under `models`. 

If a new model is trained a new model ID is created (see the output after model training). The new model will also be stored under `models`. To create the figures with the newly trained model, replace the model ID in `src/models/nn_shap.py` and `src/models/nn_scenarios.py` in the __main__ function before creating the outputs below.


## List of tables and programs

The provided code reproduces:

- [ ] All numbers provided in text in the paper
- [ ] All tables and figures in the paper
- [x] Selected tables and figures in the paper, as explained and justified below.

The code allows to reproduce all the main figures showing the model outcomes which are the bases for the results section.  


#### Calculate SHAP feature importance (Figure 4 & 5)
Run: `scl enable rh-python36 -- python src/models/nn_shap.py` 

This provide the plots for Figure 4 and 5

Figure 4 results are stored under: `reports/figures/[modelID]/shap/featureImportance_relative_[modelID]_shortTimeSuffel_.png`

Figure 5 results are stored under: 
`reports/figures/[modelID]/shap/featureImportance_time_[modelID]_shortTimeSuffel_SAU_numFeat40mask.png`
and
`reports/figures/[modelID]/shap/featureImportance_time_[modelID]_shortTimeSuffel_SAU_numFeat40mask.png`


#### Create scenario simulations  (Figures 6 & 7)
Run: `scl enable rh-python36 -- python src/models/nn_scenario.py` 

This provide the plots for Figure 6 and 7 

Individual plots are in stored folder `reports/figures/[modelID]/scenarios/`

For Figure 6 the following plots are used:
| Scenario flat rate "Sheep" | Scenario size discriminatory "Sheep" |
|-------------------|--------------------------|
| `[modelID]_flat_tprod_SAU_prev_SAU_CsubD1_SAU.png` | `[modelID]_increase_tprod_SAU_prev_SAU_CsubD1_SAU.png` |
| `[modelID]_flat_tprod_SAU_prev_SAU_DAvgSub_SAU.png` | `[modelID]_increase_tprod_SAU_prev_SAU_DAvgSub_SAU.png`  |
| `[modelID]_flat_tprod_SAU_prev_SAU_SAU.png` | `[modelID]_increase_tprod_SAU_prev_SAU_SAU.png` |

For Figure 7 the following plots are used:
| Scenario flat rate "Dairy" | Scenario size discriminatory "Dairy" |
|-------------------|--------------------------|
| `[modelID]_flat_tprod_x120_prev_x120_CsubD1_x120.png` | `[modelID]_increase_tprod_x120_prev_x120_CsubD1_x120.png` |
| `[modelID]_flat_tprod_x120_prev_x120_DAvgSub_x120.png` | `[modelID]_increase_tprod_x120_prev_x120_DAvgSub_x120.png`  |
| `[modelID]_flat_tprod_x120_prev_x120_x120.png` | `[modelID]_increase_tprod_x120_prev_x120_x120.png` |



---

## Acknowledgements

This readme was created following the guidelines from [Hindawi](https://social-science-data-editors.github.io/template_README/).