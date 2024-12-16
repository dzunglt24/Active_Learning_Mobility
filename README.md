### Install stanford CRF java library
Follow the instruction to install CRF_NER: https://nlp.stanford.edu/software/CRF-NER.html#Download

### Train CRF models for each entity type, modify prop file to specify weekly dataset
java -cp "stanford-ner.jar:lib/*" -mx4g edu.stanford.nlp.ie.crf.CRFClassifier -prop ./props/action_prop.txt

java -cp "stanford-ner.jar:lib/*" -mx4g edu.stanford.nlp.ie.crf.CRFClassifier -prop ./props/assistance_prop.txt

java -cp "stanford-ner.jar:lib/*" -mx4g edu.stanford.nlp.ie.crf.CRFClassifier -prop ./props/mobility_prop.txt

java -cp "stanford-ner.jar:lib/*" -mx4g edu.stanford.nlp.ie.crf.CRFClassifier -prop ./props/quantification_prop.txt

### Train BERT model
cd bert_model
python train.py --week 35 --entity Action

python train.py --week 35 --entity Assistance

python train.py --week 35 --entity Mobility

python train.py --week 35 --entity Quantification

### Sampling new weekly batch
python weekly_inference.py --week 35
python weekly_sampling.py --week 36
