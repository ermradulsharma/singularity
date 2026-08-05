import statistics
import csv
import io

def calculate_summary_stats(data: list) -> dict:
    """Calculates basic summary statistics for a list of numbers."""
    if not data:
        return {}
        
    return {
        "count": len(data),
        "mean": statistics.mean(data),
        "median": statistics.median(data),
        "variance": statistics.variance(data) if len(data) > 1 else 0.0,
        "std_dev": statistics.stdev(data) if len(data) > 1 else 0.0,
        "min": min(data),
        "max": max(data)
    }

def linear_regression(x: list, y: list) -> dict:
    """Calculates simple linear regression (y = mx + b)."""
    if len(x) != len(y) or len(x) < 2:
        return {"error": "x and y must have same length and at least 2 points."}
        
    try:
        slope, intercept = statistics.linear_regression(x, y)
        return {"slope": slope, "intercept": intercept, "equation": f"y = {slope:.4f}x + {intercept:.4f}"}
    except AttributeError:
        # Fallback if Python version doesn't support statistics.linear_regression
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)
        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denominator = sum((xi - x_mean)**2 for xi in x)
        
        slope = numerator / denominator if denominator != 0 else 0
        intercept = y_mean - slope * x_mean
        return {"slope": slope, "intercept": intercept, "equation": f"y = {slope:.4f}x + {intercept:.4f}"}

def parse_csv_column(csv_string: str, column_name: str) -> list:
    """Extracts a numerical column from a CSV string."""
    f = io.StringIO(csv_string)
    reader = csv.DictReader(f)
    result = []
    
    for row in reader:
        if column_name in row:
            try:
                result.append(float(row[column_name]))
            except ValueError:
                pass # Ignore non-numeric
                
    return result
