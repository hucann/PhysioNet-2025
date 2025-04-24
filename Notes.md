### To Run 
1. Create & activate virtual environment (via requirements.txt)
```bash
conda activate physionet
```
2. Download & place data folder
data_train
data_test

3. Train model 
```bash 
python train_model.py -d data_train -m model -v
```
2. Run model
```bash
python run_model.py -d data_test -m model -o outputs -v
```
- Evaluate model
```bash
python evaluate_model.py -d data_test -o outputs -s scores.csv
```

---
### Dataset 
CODE-15
- 300,000 
- Weak label: may not be validated (self reported) (positive rate = 6562/343425 = 1.91%)
- 400 Hz; 7.3 s or 10.2 s

SaMi-Trop
- 1,631
- Strong label: validated by serological tests -- all **positive**
- 400 Hz; 7.3 s or 10.2 s

PTB-XL
- 21,799
- Strong label: all or almost all likely to be Chagas **negative** based on geography
- 500 Hz; 10s

To Note:
- The prevalence rate of Chagas disease in each of the training, validation, and test sets approximately matches the prevalence rate of the countries in which Chagas disease is endemic.
-  1%; 3% ~ 6% 

---
### Task
Detection of Chagas Disease from ECG 
(Develop automated approaches for addressing important physiological & clinical problems)

### Background
Why ECG
- Chagas disease is a parasitic disease in Central and South America.
- Serological testing can detect Chagas disease, but testing capacity is limited.
- Chagas disease symptoms may also appear in electrocardiograms (ECG), so ECGs can help to prioritize individuals for the limited numbers of serological tests

Chagas disease
- Acute phase & life-long chronic phase 
- In the early stages of infection, Chagas disease has no or mild symptoms, and can be treated by specific drugs that can prevent the progression of the disease

