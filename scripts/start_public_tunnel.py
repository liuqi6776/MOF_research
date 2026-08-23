"""
Production Public Tunnel Launcher for MOF Chatbot
使用 Localtunnel 将本地 7860 端口发布至公网 HTTPS (Custom Subdomain: mof-chatbot.loca.lt)
"""
import sys
import io
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass

import os
import re
import time
import json
import urllib.request
import subprocess

def get_public_ip():
    """获取外网出口 IP (用于 Localtunnel 首次访问的安全验证密码)"""
    for endpoint in ['https://ipv4.icanhazip.com', 'https://api.ipify.org', 'https://ifconfig.me/ip']:
        try:
            req = urllib.request.Request(endpoint, headers={'User-Agent': 'Mozilla/5.0'})
            ip = urllib.request.urlopen(req, timeout=4).read().decode('utf-8').strip()
            if ip:
                return ip
        except Exception:
            continue
    return "172.96.161.31"

def start_public_tunnel():
    public_ip = get_public_ip()
    print(f"[*] Server Public IP: {public_ip}", flush=True)
    print("[*] Launching Public HTTPS Tunnel for MOF Chatbot (Port: 7860)...", flush=True)
    
    # 优先使用专属自定义二级域名 mof-chatbot
    cmd = ['cmd.exe', '/c', 'npx', 'localtunnel', '--port', '7860', '--subdomain', 'mof-chatbot']
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding='utf-8',
        errors='ignore'
    )
    
    public_url = None
    url_pattern = re.compile(r'https://[a-zA-Z0-9-]+\.loca\.lt')
    
    start_time = time.time()
    while time.time() - start_time < 20:
        line = process.stdout.readline()
        if not line:
            time.sleep(0.5)
            continue
            
        print(f"  [Tunnel] {line.strip()}", flush=True)
        match = url_pattern.search(line)
        if match:
            public_url = match.group(0)
            break
            
    if not public_url:
        # 备用方案：随机域名
        print("[*] Retrying with dynamic subdomain...", flush=True)
        cmd_fallback = ['cmd.exe', '/c', 'npx', 'localtunnel', '--port', '7860']
        process = subprocess.Popen(
            cmd_fallback,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding='utf-8',
            errors='ignore'
        )
        start_time = time.time()
        while time.time() - start_time < 20:
            line = process.stdout.readline()
            if not line:
                time.sleep(0.5)
                continue
            match = url_pattern.search(line)
            if match:
                public_url = match.group(0)
                break

    if public_url:
        print("\n" + "="*80, flush=True)
        print("  🎉 MOF Chatbot 公网访问地址已成功发布 / Public Access Live:", flush=True)
        print(f"  👉 公网 HTTPS 地址: {public_url}", flush=True)
        print(f"  👉 访问验证密码 (Tunnel Password / Endpoint IP): {public_ip}", flush=True)
        print(f"  👉 本地访问地址:   http://127.0.0.1:7860", flush=True)
        print("="*80 + "\n", flush=True)
        
        os.makedirs("results", exist_ok=True)
        with open("results/public_url.txt", "w", encoding="utf-8") as f:
            f.write(f"Public URL: {public_url}\nPassword/IP: {public_ip}\nLocal URL: http://127.0.0.1:7860\n")
            
        # 持续常驻运行并自动保活
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            process.terminate()
    else:
        print("[!] Failed to establish public tunnel.", flush=True)

if __name__ == "__main__":
    start_public_tunnel()
