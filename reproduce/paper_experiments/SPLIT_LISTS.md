The train/valid/test id lists referenced by the launchers (configs/illust/{train,valid,test}_paper.txt) are identical to
reproduce/data/splits/{train,valid,test}.txt (= detfill/configs/illust/*.txt) and were de-duplicated; recreate them with
  for s in train valid test; do cp reproduce/data/splits/$s.txt <configs>/illust/${s}_paper.txt; done
The natural-image (ImageNet) lists remain under segmenter_dependency/configs/real/.
