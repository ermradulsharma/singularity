import math

def forward_kinematics_2d(l1: float, l2: float, theta1: float, theta2: float) -> tuple:
    """Calculates End-Effector position for a 2D robotic arm"""
    x = l1 * math.cos(theta1) + l2 * math.cos(theta1 + theta2)
    y = l1 * math.sin(theta1) + l2 * math.sin(theta1 + theta2)
    return round(x, 4), round(y, 4)

def inverse_kinematics_2d(l1: float, l2: float, x: float, y: float) -> tuple:
    """Calculates joint angles required to reach (x,y)"""
    cos_theta2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    if cos_theta2 < -1 or cos_theta2 > 1:
        return None # Unreachable
    theta2 = math.acos(cos_theta2)
    theta1 = math.atan2(y, x) - math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    return round(theta1, 4), round(theta2, 4)
