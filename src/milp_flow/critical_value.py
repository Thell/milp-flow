import math


def find_critical_cost_c(A: float, B: float, c0: float, c1: float, budget: float = math.inf) -> float:
    """
    Calculates the critical value of the shared cost 'C' at which the
    effective value ratio of Prize B and Prize A become equal.

    The problem is defined by the inequality: B / (c1 + C) > A / (c0 + C)

    The critical point C_star is found when the ratios are equal:
    C_star = (A * c1 - B * c0) / (B - A)

    Args:
        A (float): The base value of Prize A.
        B (float): The base value of Prize B.
        c0 (float): The direct cost associated with Prize A.
        c1 (float): The direct cost associated with Prize B.
        budget (float): The maximum allowed value for C. Defaults to infinity.

    Returns:
        float: The calculated critical cost C, constrained by 0 and the budget.
               Raises a ValueError if B <= A.
    """
    if B <= A:
        # Constraint B > A must be satisfied for a meaningful critical point calculation
        # and to avoid division by zero or a reversal of the inequality logic.
        raise ValueError("The value of Prize B must be greater than Prize A (B > A).")

    # Calculate the critical value C_star where the ratios are exactly equal
    numerator = (A * c1) - (B * c0)
    denominator = B - A

    C_star = numerator / denominator

    # The cost C is constrained by 0 <= C <= Budget.

    # 1. C must be non-negative
    C_constrained = max(0.0, C_star)

    # 2. C must not exceed the budget
    C_constrained = min(budget, C_constrained)

    # The printed details now include c0 and c1
    print(f"--- Calculation Details ---")
    print(f"Prize A Value: {A} (Direct Cost c0={c0})")
    print(f"Prize B Value: {B} (Direct Cost c1={c1})")
    print(f"Budget Limit: {budget if budget != math.inf else 'Infinite'}")
    print(f"Critical Point (C_star, point of equality): {C_star:.4f}")
    print("---------------------------")

    return C_constrained


# --- Example Usage (using c0=1 and c1=3 as specified in the original problem) ---
C0_VAL = 1.0
C1_VAL = 3.0

print("\n--- Example 1: C_star falls within the [0, Budget] range (A=10, B=20, c0=1, c1=3) ---")
# C_star = (10*3 - 20*1) / (20 - 10) = 10 / 10 = 1.0
A1 = 10.0
B1 = 20.0
Budget1 = 5.0
critical_C1 = find_critical_cost_c(A1, B1, C0_VAL, C1_VAL, Budget1)
print(f"Result 1: Critical C is {critical_C1:.2f}\n")

print("--- Example 2: C_star is below 0 (A=10, B=35, c0=1, c1=3) ---")
# C_star = (10*3 - 35*1) / (35 - 10) = -5 / 25 = -0.2
A2 = 10.0
B2 = 35.0
Budget2 = 10.0
critical_C2 = find_critical_cost_c(A2, B2, C0_VAL, C1_VAL, Budget2)
print(f"Result 2: Critical C is {critical_C2:.2f}\n")

print("--- Example 3: C_star exceeds the Budget (A=10, B=12, c0=1, c1=3) ---")
# C_star = (10*3 - 12*1) / (12 - 10) = 18 / 2 = 9.0
A3 = 10.0
B3 = 12.0
Budget3 = 5.0
critical_C3 = find_critical_cost_c(A3, B3, C0_VAL, C1_VAL, Budget3)
print(f"Result 3: Critical C is {critical_C3:.2f}\n")

print("--- Example 4: No Budget constraint (A=10, B=20, c0=1, c1=3) ---")
# C_star = (10*3 - 20*1) / (20 - 10) = 1.0
A4 = 10.0
B4 = 20.0
critical_C4 = find_critical_cost_c(A4, B4, C0_VAL, C1_VAL)
print(f"Result 4: Critical C is {critical_C4:.2f}\n")

print("--- Example 5: Using different direct costs (A=50, B=70, c0=5, c1=10) ---")
# A=50, B=70. C_star = (50*10 - 70*5) / (70 - 50) = (500 - 350) / 20 = 150 / 20 = 7.5
A5 = 50.0
B5 = 70.0
C0_EX5 = 5.0
C1_EX5 = 10.0
Budget5 = 20.0
critical_C5 = find_critical_cost_c(A5, B5, C0_EX5, C1_EX5, Budget5)
print(f"Result 5: Critical C is {critical_C5:.2f}\n")

# Example 6: Violation of the B > A constraint
# try:
#     find_critical_cost_c(A=10.0, B=10.0, c0=1.0, c1=3.0, budget=5.0)
# except ValueError as e:
#     print(f"Handled error: {e}")

print(" --- Real World Example ---")
# A=1516377, B=3816445. C_star = (100*3 - 200*1) / (200 - 100) = 100 / 100 = 1.0
A6 = 1516377.0
B6 = 3816445.0
C0_EX6 = 1.0
C1_EX6 = 3.0
critical_C6 = find_critical_cost_c(A6, B6, C0_EX6, C1_EX6)
print(f"Result 6: Critical C is {critical_C6:.2f}\n")
