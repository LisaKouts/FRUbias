# FairFRU 

FairFRU ([see here](https://pypi.org/project/FairFRU/)) is a Python package providing algorithms able to quantify fairness in automated decision making using the fuzzy-rough set theory. 

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install FairFRU.

```bash
pip install FairFRU
```
## Usage
```python
import FairFRU

# create toy dataset
df = pd.DataFrame([
    [2200,4200,6000,'male','house',1],
    [7200,2600, 7600,'male','rent',1],
    [3900,3600, 8150,'female','rent',0],
    [3900,3600, 8150,'non-binary','rent',1],
    [10400,3900,9100,'female','rent',0],
    [8300,2500,4300,'non-binary','rent',0]], 
    columns=['debt','salary','portfolio','gender','housing','creditworthy'])

# specify conditions, classication target and path to save files
target = 'creditworthy'
c = np.array(['debt','salary','portfolio','gender','housing'])
path = r"C:\Path\to\my\folder"

# create folder to store the membership values to the fuzzy-rough regions
path = os.path.join(path, 'bank_toy')
if not os.path.exists(path):
    os.makedirs(os.path.join(path))

# compute membership values to the fuzzy-rough positive, negative and boundary regions for the complete dataset and after removing each feature from the data
FRU(df, path).membership_values(c, target, 'bank', similarity='HVDM', hide_progress=True)

# load the membership values
values = FRU(df,path).load_membership_values()

# retrieve values when the complete set of features is used
POS, NEG, BND = values['full']

# observe values for decision class 1
pd.DataFrame([POS[1], NEG[1], BND[1]], index=['POS', 'NEG', 'BND']).T
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
