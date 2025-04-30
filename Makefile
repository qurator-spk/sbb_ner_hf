OPT_PARAMETERS=--batch-size 32 --batch-size 64 --batch-size 96 --learning-rate 2e-5 --learning-rate 4e-5 --learning-rate 1e-4

single:
	mkdir -p models-single
	python experiment.py results-single.pkl --model-storage-path models-single --max-epochs=30 --exp-type single --use-test-config all $(OPT_PARAMETERS)

historical:
	python experiment.py results-historical.pkl --max-epochs=30 --exp-type historical $(OPT_PARAMETERS) 

contemporary:
	python experiment.py results-contemporary.pkl --max-epochs=30 --exp-type contemporary $(OPT_PARAMETERS) 

zefys-pretrain:
	mkdir -p zefys-pretrained
	python experiment.py zefys-pretrained/pretrained-on-zefys-configs.pkl --model-storage-path zefys-pretrained --use-data-config zefys2025 --max-epochs=30

historic-on-zefys-pretrained:
	mkdir -p on-zefys-pretrained
	python experiment.py --model-storage-path on-zefys-pretrained --pretrain-path=zefys-pretrained --pretrain-config-file=zefys-pretrained/pretrained-on-zefys-configs.pkl results-pretrained-zefys.pkl --use-data-config "europeana-lft" --use-data-config "europeana-onb" --use-data-config "hipe2020" --use-data-config "hisgerman" --use-data-config "neiss-arendt" --use-data-config "neiss-sturm" $(OPT_PARAMETERS) 


historic-all-pretrain:
	mkdir -p all-historic-pretrained
	python experiment.py all-historic-pretrained/pretrained-on-all-historic-configs.pkl --model-storage-path all-historic-pretrained --use-data-config all-historic --max-epochs=30

historic-on-all-historic-pretrained:
	mkdir -p on-all-historic-pretrained
	python experiment.py --model-storage-path on-all-historic-pretrained --pretrain-path=all-historic-pretrained --pretrain-config-file=all-historic-pretrained/pretrained-on-all-historic-configs.pkl results-pretrained-all-historic.pkl --exp-type historical $(OPT_PARAMETERS) 

all:	historical contemporary zefys-pretrain historic-on-zefys-pretrained all-historic-pretrain historic-on-all-historic--pretrained

neiss-sturm:
	python experiment.py results_neiss-sturm.pkl --use-data-config neiss-sturm --max-epochs=30 

conll2003:
	python experiment.py connll2003.pkl --use-data-config conll2003 --max-epochs=30 
