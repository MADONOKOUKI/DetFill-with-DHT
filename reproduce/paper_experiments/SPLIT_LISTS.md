The train/valid/test id lists referenced by the launchers (`configs/illust/{train,valid,test}_paper.txt`, also under
`diffusart_retrain/code/configs/illust/` together with the 200-image subset `test_paper_sub200.txt`) contain the same
20,000 / 3,000 / 3,000 ids as `reproduce/data/splits/{train,valid,test}.txt` (= `detfill/configs/illust/*.txt`), written as
relative `<bucket>/<id>.image.png` lines. The natural-image (ImageNet) lists are under `segmenter_dependency/configs/real/`
and `detfill/configs/real/`.
