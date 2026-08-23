"""
Production MOF Chatbot Public Server & Tunnel Launcher
一键启动 MOF Chatbot Web 界面并发布到全网可访问的专属英文 HTTPS 域名
"""
import os
import sys
import time
import json
import urllib.request
import subprocess

NGROK_PATH = r"C:\Users\liuqi\AppData\Local\Microsoft\WindowsApps\ngrok.EXE"

def run_service():
    print("[1/2] 正在启动 MOF Chatbot 本地服务 (Port: 7860)...", flush=True)
    app_process = subprocess.Popen(
        [sys.executable, "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding='utf-8',
        errors='ignore'
    )
    
    # 等待本地 7860 端口启动
    local_ready = False
    for _ in range(20):
        time.sleep(1)
        try:
            res = urllib.request.urlopen("http://127.0.0.1:7860", timeout=2)
            if res.status == 200:
                local_ready = True
                break
        except Exception:
            continue
            
    if not local_ready:
        print("[!] 警告: 本地服务启动超时，请检查 app.py 日志。", flush=True)
        return
        
    print("[2/2] 正在建立公网 HTTPS 穿透通道 (英文专属域名)...", flush=True)
    ngrok_process = subprocess.Popen(
        [NGROK_PATH, "http", "7860"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    
    public_url = None
    for _ in range(15):
        time.sleep(1)
        try:
            res = urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=2)
            data = json.loads(res.read().decode('utf-8'))
            tunnels = data.get("tunnels", [])
            if tunnels:
                public_url = tunnels[0].get("public_url")
                if public_url:
                    break
        except Exception:
            continue
            
    if public_url:
        print("\n" + "="*85, flush=True)
        print("  [+] MOF Chatbot Public Live URL Active:", flush=True)
        print(f"  [+] Public URL: {public_url}", flush=True)
        print(f"  [+] Local URL:  http://127.0.0.1:7860", flush=True)
        print("="*85 + "\n", flush=True)
        print("[*] Public access is live. Open the Public URL to interact with 3D crystal models & AI!\n", flush=True)
        
        os.makedirs("results", exist_ok=True)
        with open("results/public_url.txt", "w", encoding="utf-8") as f:
            f.write(f"Public Live URL: {public_url}\nLocal URL: http://127.0.0.1:7860\n")
            
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            app_process.kill()
            ngrok_process.kill()
    else:
        print("[!] 未能获取 Ngrok 公网地址，请检查服务状态。", flush=True)

if __name__ == "__main__":
    run_service()
