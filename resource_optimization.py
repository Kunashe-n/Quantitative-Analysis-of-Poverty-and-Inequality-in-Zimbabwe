import pulp

def run_lp_optimization():
    # Define the problem: We want to minimize poverty
    model = pulp.LpProblem("Poverty_Minimization", pulp.LpMinimize)

    # Decision variables: % allocation to each sector (0 to 100%)
    x1 = pulp.LpVariable('Education', lowBound=0, upBound=100)
    x2 = pulp.LpVariable('Healthcare', lowBound=0, upBound=100)
    x3 = pulp.LpVariable('CashTransfers', lowBound=0, upBound=100)

    # Constraint: Total budget must equal 100%
    model += x1 + x2 + x3 == 100, "Total_Budget_Constraint"

    # Objective function: Minimize poverty headcount 
    # Baseline assumed at 20% for this sub-model, with reduction coefficients
    model += 0.20 - 0.048*(x1/100) - 0.023*(x2/100) - 0.030*(x3/100), "Objective"

    # Solve the model
    model.solve()

    # Output Results
    print("--- Linear Programming Optimization Results ---")
    print(f"Status: {pulp.LpStatus[model.status]}")
    print("\nOptimal Allocation:")
    print(f"Education: {x1.varValue} %")
    print(f"Healthcare: {x2.varValue} %")
    print(f"Cash Transfers: {x3.varValue} %")
    print(f"\nMinimized poverty rate: {pulp.value(model.objective) * 100:.2f} %")

if __name__ == "__main__":
    run_lp_optimization()
