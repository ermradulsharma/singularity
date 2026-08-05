def taylor_tool_life(v: float, n: float, C: float) -> float:
    """Production Tech: Taylor's Tool Life Equation (V * T^n = C), returns Tool Life (T)"""
    return (C / v) ** (1 / n)

def machining_time(length: float, feed: float, rpm: float) -> float:
    """Manufacturing Process: Lathe Machining Time calculation"""
    return length / (feed * rpm)

def pid_controller(kp, ki, kd, error, integral, derivative):
    """Mechatronics: Basic PID Controller logic simulation"""
    return (kp * error) + (ki * integral) + (kd * derivative)
