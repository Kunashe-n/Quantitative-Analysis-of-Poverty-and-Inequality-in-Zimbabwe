import pandas as pd
import statsmodels.api as sm

def run_inequality_regression():
    # Example dataset (replace with real Zimbabwe National Household Survey data)
    data = pd.DataFrame({
        'income': [100, 200, 300, 400, 1000],
        'education': [0, 1, 1, 2, 3],  # coded levels: 0=Primary, 3=Tertiary
        'sector': [0, 1, 1, 1, 0],     # 0=Informal, 1=Formal
        'region': [1, 1, 0, 0, 0]      # 1=Rural, 0=Urban
    })

    # Define independent variables (X) and dependent variable (y)
    X = data[['education', 'sector', 'region']]
    X = sm.add_constant(X)  # Adds the intercept term
    y = data['income']

    # Fit the OLS model
    model = sm.OLS(y, X).fit()
    
    # Output the results
    print("--- OLS Regression Results ---")
    print(model.summary())

if __name__ == "__main__":
    run_inequality_regression()
