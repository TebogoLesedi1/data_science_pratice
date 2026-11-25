# Melbourne House Price Prediction & Exploratory Analysis

An exploratory data analysis (EDA) and machine learning pipeline built in Python to analyze housing market trends in Melbourne and predict property prices using the `MELBOURNE_HOUSE_PRICES_LESS.csv` dataset.

## 📊 Dataset Overview
The dataset originally contains **63,023 records** and **13 features** covering Melbourne property listings:
* **Location Features:** `Suburb`, `Address`, `Postcode`, `Regionname`, `CouncilArea`, `Distance` (km from CBD)
* **Property Details:** `Rooms`, `Type` (`h` = house, `t` = townhouse, `u` = unit/apartment), `Propertycount`
* **Transaction Details:** `Price` (target variable), `Method` (sale method), `SellerG` (real estate agency), `Date`

## 🛠️ Data Preprocessing & Pipeline
1. **Data Cleaning:** Removed **14,590 records** with missing `Price` values via `.dropna()`, leaving **48,433 complete rows**. Checked for duplicate entries.
2. **Exploratory Visualizations:**
   * **Sale Method Distribution:** Pie chart breaking down property sales across auction and private sale methods (`Method`).
   * **Agency Activity:** Line plot tracking listing volume across real estate agencies (`SellerG`).
   * **Rooms vs. Average Price:** Bar chart analyzing mean property prices across room counts (filtered to `Rooms <= 8` to exclude extreme outliers).
3. **Feature Engineering & Encoding:**
   * Cleaned column headers and applied `LabelEncoder` across all 8 categorical (`object`) features (`Suburb`, `Address`, `Type`, `Method`, `SellerG`, `Date`, `Regionname`, `CouncilArea`).
4. **Train/Test Split & Modeling:**
   * Split features (`X`) and target (`Price`) into training and testing sets using `train_test_split`.
   * Evaluated regression and classification baselines from `scikit-learn` (`LinearRegression`, `KNeighborsClassifier`, `SVC`, `GaussianNB`, `RandomForestClassifier`, `DecisionTreeClassifier`, `GradientBoostingClassifier`).

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3 installed along with the following libraries:
```bash
pip install pandas numpy matplotlib scikit-learn
