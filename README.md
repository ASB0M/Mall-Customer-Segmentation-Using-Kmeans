# Mall Customer Segmentation

A machine learning project applying K-Means clustering to segment mall customers based on demographic and behavioural data.

## Overview

This project uses unsupervised learning to identify distinct customer segments from a mall customer dataset. By grouping customers with similar characteristics, the analysis can inform targeted marketing strategies and business decisions.

## Dataset

**File:** `Mall Customers.csv`

**Features:**
- `CustomerID` — Unique customer identifier (dropped before modelling)
- `Gender` — Male / Female
- `Age` — Customer age
- `Education` — Education level (Uneducated → Doctorate)
- `Marital Status` — Divorced / Married / Single / Unknown
- `Annual Income (k$)` — Annual income in thousands of dollars
- `Spending Score (1-100)` — Mall-assigned score based on spending behaviour

## Methodology

1. **Exploratory Data Analysis** — Distribution of income, spending score, and education across genders
2. **Preprocessing**
   - One-Hot Encoding for `Gender` and `Marital Status`
   - Ordinal Encoding for `Education` (ordered scale)
   - Dropped `CustomerID`
   - StandardScaler applied to all features before clustering
3. **Clustering** — K-Means with `k-means++` initialisation; optimal K selected via the Elbow Method (K=5)
4. **Visualisation** — Cluster scatter plots and a 2D PCA projection

## Tech Stack

- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## Project Structure

```
├── Mall_Customer_Segmentation.ipynb   # Main notebook
├── Mall Customers.csv                 # Dataset
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook Mall_Customer_Segmentation.ipynb
```

## Results

K-Means identified **5 distinct customer segments** based on income, spending behaviour, age, and demographic features. The PCA projection confirms well-separated clusters in reduced 2D space.
