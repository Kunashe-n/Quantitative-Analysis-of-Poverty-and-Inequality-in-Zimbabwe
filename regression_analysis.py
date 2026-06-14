import pandas as pd
import numpy as np
import statsmodels.api as sm

def generate_synthetic_zimbabwe_data(n_households=5000):
    print("Generating synthetic microdata calibrated to Zimbabwe national statistics...")
    np.random.seed(42) # Ensures reproducibility
    
    # Generate independent variables based on known demographic splits in Zimbabwe
    # Geography: ~67% Rural (1), 33% Urban (0)
    is_rural = np.random.choice([1, 0], size=n_households, p=[0.67, 0.33])
    
    # Employment: ~20% Formal (1), 80% Informal/Unemployed (0)
    is_employed_formal = np.random.choice([1, 0], size=n_households, p=[0.20, 0.80])
    
    # Education: 0=Primary, 1=Secondary, 2=Tertiary
    # Probabilities shift slightly based on rural vs urban
    education_level = np.where(
        is_rural == 1,
        np.random.choice([0, 1, 2], size=n_households, p=[0.50, 0.45, 0.05]),
        np.random.choice([0, 1, 2], size=n_households, p=[0.10, 0.60, 0.30])
    )

    # Base income generation (log-normal distribution)
    base_income = np.random.lognormal(mean=1.5, sigma=0.5, size=n_households)
    
    # Apply the structural determinants (coefficients)
    # 1. Rural penalty (decreases income)
    # 2. Formal employment premium (increases income)
    # 3. Education premium (Tertiary yields highest multiplier)
    income = (base_income 
              * np.where(is_rural == 1, 0.82, 1.0) 
              * np.where(is_employed_formal == 1, 1.25, 1.0)
              * np.where(education_level == 2, 3.5, np.where(education_level == 1, 1.5, 1.0)))
    
    # Add some random noise to simulate real-world variance
    income = income + np.random.normal(0, 0.5, size=n_households)
    income = np.maximum(income, 0.5) # Prevent negative incomes
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame({
        'household_income_usd': income,
        'education_level': education_level,
        'is_employed_formal': is_employed_formal,
        'is_rural': is_rural
    })
    
    df.to_csv('survey_data.csv', index=False)
    print("-> Successfully saved 'survey_data.csv' (5000 rows).")
    return df

def run_inequality_regression():
    # 1. Generate or load the dataset
    data = generate_synthetic_zimbabwe_data()

    # 2. Define variables for OLS Regression
    X = data[['education_level', 'is_employed_formal', 'is_rural']]
    X = sm.add_constant(X)
    
    # Using log of income is standard practice in econometric poverty models
    y = np.log(data['household_income_usd'])

    # 3. Fit the model
    model = sm.OLS(y, X).fit()
    
    print("\n--- OLS Regression Results ---")
    print(model.summary())

if __name__ == "__main__":
    run_inequality_regression()
