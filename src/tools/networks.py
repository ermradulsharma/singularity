def calculate_subnet_mask(cidr: int) -> str:
    """Network Subnet Mask Calculation from CIDR notation"""
    mask = (0xffffffff >> (32 - cidr)) << (32 - cidr)
    return f"{(mask >> 24) & 0xff}.{(mask >> 16) & 0xff}.{(mask >> 8) & 0xff}.{mask & 0xff}"

def calculate_network_address(ip: str, cidr: int) -> str:
    """Calculates Network ID given an IP and CIDR"""
    ip_parts = [int(p) for p in ip.split('.')]
    ip_bin = (ip_parts[0] << 24) + (ip_parts[1] << 16) + (ip_parts[2] << 8) + ip_parts[3]
    mask = (0xffffffff >> (32 - cidr)) << (32 - cidr)
    net_bin = ip_bin & mask
    return f"{(net_bin >> 24) & 0xff}.{(net_bin >> 16) & 0xff}.{(net_bin >> 8) & 0xff}.{net_bin & 0xff}"
