## 🧩 Environment Setup and Code Test 🚀

To reproduce the environment used for **SphereKriging** and run test code, follow the steps below.

### 1️⃣ Clone the repository
```
git clone https://github.com/khrisyu9/SphereKriging.git
cd SphereKriging
```
### 2️⃣ Create a new Conda environment
```
conda create -n SphereKriging python=3.11
conda activate SphereKriging
```
### 3️⃣ Install dependencies
```
pip install -r requirements.txt
```
### 4️⃣ Run different tests
Baseline DeepKriging on Spherical Data:
```
python DeepKriging_Sim.py
```
Spherical DeepKriging with Great-Circle RBFs:
```
### Spherical DeepKriging with Great-Circle RBFs 
python SDK_RBF_Sim.py
```
Spherical DeepKriging with Spherical CNN:
```
python SDK_SCNN_Sim.py
```
