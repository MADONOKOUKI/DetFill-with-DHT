# Supp. Table — segmentation-dependency cross-evaluation (train seg. × eval seg.), MSE/LPIPS/DreamSim Hint-AUC, mean ± SD over 3 sketch sources

| train seg. | eval seg. | MSE | LPIPS | DreamSim | check |
|---|---|---|---|---|---|
| DanbooRegion | Felzenszwalb | 0.0160±0.0045 | 0.1960±0.0469 | 0.0943±0.0221 | OK |
| DanbooRegion | DanbooRegion | 0.0172±0.0049 | 0.2045±0.0470 | 0.0990±0.0223 | OK |
| DanbooRegion | SLIC | 0.0269±0.0044 | 0.2756±0.0540 | 0.1554±0.0331 | OK |
| SLIC | Felzenszwalb | 0.0203±0.0059 | 0.2182±0.0445 | 0.1012±0.0189 | OK |
| SLIC | DanbooRegion | 0.0209±0.0061 | 0.2321±0.0487 | 0.1104±0.0209 | OK |
| SLIC | SLIC | 0.0269±0.0063 | 0.2370±0.0436 | 0.1214±0.0214 | OK |
| Felzenszwalb | Felzenszwalb† | 0.0149±0.0050 | 0.1829±0.0472 | 0.0839±0.0212 | OK |
| Felzenszwalb | DanbooRegion | 0.0164±0.0051 | 0.1999±0.0487 | 0.0945±0.0225 | OK |
| Felzenszwalb | SLIC | 0.0243±0.0049 | 0.2615±0.0563 | 0.1397±0.0324 | OK |
