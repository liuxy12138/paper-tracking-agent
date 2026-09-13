# 用自己的 Windows 电脑免费分享

你的电脑就是“服务器环境”：项目、Milvus、Redis 和模型都在这台电脑上运行。Tailscale Funnel 提供一个 `https://设备名.网络名.ts.net/` 形式的公网地址，不需要购买域名、固定公网 IP 或做路由器端口映射。

## 1. 启动本地依赖

先打开 Docker Desktop，并确认它显示 Engine running。在 PowerShell 中运行：

```powershell
cd D:\agent1
.\standalone.bat start
docker run -d --name research-redis -p 6379:6379 redis:7
```

如果 Redis 容器之前已创建，改用 `docker start research-redis`。

## 2. 启动受密码保护的网站

```powershell
cd D:\agent1
python -m pip install -r requirements.txt
if (-not (Test-Path agent_config.json)) { Copy-Item agent_config.example.json agent_config.json }
$env:ZHIPU_API_KEY = "你的智谱 API Key"
$env:SHARE_USERNAME = "researcher"
$env:SHARE_PASSWORD = "你自己设置的长密码"
python agent_main.py check-milvus
python -m uvicorn competitive_research_agent.webapp:app --host 127.0.0.1 --port 8000
```

保持这个 PowerShell 窗口运行。先用本机浏览器打开 `http://127.0.0.1:8000/`，确认密码提示和页面正常。实际问答前需要导入资料；首次启动可能下载 BGE 模型。不要把智谱 Key 发给其他人。

## 3. 取得免费 HTTPS 公网链接

安装并登录 [Tailscale](https://tailscale.com/download/windows)，启用账号的 MagicDNS、HTTPS 与 Funnel。另开一个 PowerShell 窗口执行：

```powershell
tailscale funnel 8000
```

首次运行可能引导你在浏览器里批准 Funnel。命令输出的 `https://…ts.net/` 就是分享给朋友的链接；朋友无需安装 Tailscale，但需输入你设置的访问用户名和密码。窗口关闭或电脑休眠后，分享服务可能中断。停止分享可运行 `tailscale funnel reset`。

**费用边界：**Tailscale Personal 目前提供免费方案；你的电脑和网络要持续运行。智谱 GLM 的真实问答仍可能产生 API Token 费用。这个共享密码适合少量认识的人试用，不是开放注册的多租户系统。
