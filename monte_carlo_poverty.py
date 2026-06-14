import numpy as np

def run_monte_carlo_simulation():
    # Base Parameters
    n_households = 10000
    poverty_line = 3.0  # $3/day poverty line
    
    # Generate baseline income distribution (lognormal distribution common for income)
    baseline_income = np.random.lognormal(mean=1.0, sigma=0.5, size=n_households)
    baseline_poverty = np.mean(baseline_income < poverty_line)

    # Inflation shock parameters
    inflation_mean = 0.10  # 10% average food inflation
    inflation_std = 0.02   # 2% volatility

    # Monte Carlo simulation
    n_simulations = 1000
    poverty_rates = []

    for _ in range(n_simulations):
        # Simulate an inflation shock for this iteration
        shock = np.random.normal(inflation_mean, inflation_std)
        new_income = baseline_income * (1 - shock) # Real income drops as inflation rises
        
        # Calculate new poverty rate
        poverty_rate = np.mean(new_income < poverty_line)
        poverty_rates.append(poverty_rate)

    # Calculate final statistics
    mean_poverty = np.mean(poverty_rates)
    std_poverty = np.std(poverty_rates)

    # Output Results
    print("--- Monte Carlo Simulation Results ---")
    print(f"Baseline poverty rate: {baseline_poverty * 100:.2f}%")
    print(f"Post-shock poverty rate (mean): {mean_poverty * 100:.2f}%")
    print(f"Standard deviation across simulations: {std_poverty * 100:.2f}%")

if __name__ == "__main__":
    run_monte_carlo_simulation()
