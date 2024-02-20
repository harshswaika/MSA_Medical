# MSA_Medical

## Victim models:
1. GBCNet (/home/ankita/mnt/data_msa_medical/victim_models/gbcnet.pth)
2. Radformer (/home/ankita/mnt/data_msa_medical/victim_models/radformer/radformer.pkl)

## Datasets:
1. GBC aka gbusg - victim dataset
2. GBUSV - thief dataset

## ActiveThief: 
Use the script `activethief_gbc.py` to train a thief model using random sample selection. Config files are named as `<victim_arch>_<thief_arch>_<thief_dataset>.yaml`
```
cd activethief
python activethief_gbc.py --c configs/gbc/radformer_resnet50_gbusv.yaml
```