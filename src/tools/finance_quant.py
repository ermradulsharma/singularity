import math

def calculate_cagr(start_value: float, end_value: float, years: float) -> float:
    """Calculates Compound Annual Growth Rate (CAGR)"""
    if start_value <= 0 or years <= 0:
        return 0.0
    return (math.pow(end_value / start_value, 1 / years) - 1) * 100

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculates the European Call Option price using Black-Scholes model.
    S: Current stock price
    K: Strike price
    T: Time to expiration (in years)
    r: Risk-free interest rate (annualized)
    sigma: Volatility of the stock (annualized)
    """
    if T <= 0:
        return max(0.0, S - K)
        
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    # Cumulative standard normal distribution approximation
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
        
    call_price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return call_price

def calculate_npv(rate: float, cash_flows: list) -> float:
    """
    Calculates Net Present Value (NPV).
    rate: discount rate
    cash_flows: list of cash flows, where index is the period (Year 0, 1, 2...)
    """
    npv = sum(cf / (1 + rate)**i for i, cf in enumerate(cash_flows))
    return npv
