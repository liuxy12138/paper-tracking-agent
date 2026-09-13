# 公网 HTTPS 部署

此目录将 FastAPI、Milvus、Redis 和 Caddy 放在同一个 Docker Compose 网络中。公网只开放 Caddy 的 80/443 端口；Milvus、Redis 和应用的 8000 端口不直接对外暴露。Caddy 使用域名自动申请与续签 HTTPS 证书，并用访问密码保护页面、API 和 SSE。

## 准备

- 一台能运行 Docker Compose 的公网 Linux 服务器。Milvus 和 BGE 模型需要足够的内存、磁盘和可访问模型下载站点的网络。
- 一个由你控制的域名，例如 `research.example.com`。将其 A/AAAA 记录指向服务器，并在防火墙和云安全组开放 TCP 80、443。
- 智谱 API Key。真实问答会消耗该账户的 GLM Token。

## 首次启动

在服务器上克隆本仓库，进入 `agent1/deploy`：

```bash
cp .env.example .env
docker run --rm caddy:2 caddy hash-password --plaintext '你准备分享的长密码'
```

把生成的哈希填入 `.env` 的 `BASIC_AUTH_HASH`，保留单引号；填入真实的 `DOMAIN`、`BASIC_AUTH_USER` 和 `ZHIPU_API_KEY`。不要将 `.env` 提交到 Git。

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f caddy app
```

首次构建和 BGE 模型下载可能需要较长时间。服务就绪后，通过 `https://你的域名/` 打开前端；浏览器会要求输入设置的用户名和密码。API 文档在 `https://你的域名/docs`。可以将域名和访问凭据分别发给获准使用的人。

## 后续操作

```bash
docker compose up -d --build
docker compose logs --tail=100 app
docker compose down
```

`docker compose down` 不删除命名数据卷。更新代码后重新构建应用。若要更换密码，重新生成哈希、修改 `.env`，然后执行 `docker compose up -d`。

这套配置是受密码保护的共享实例。每次打开页面会生成独立会话 ID，但它不是完整的多用户账号和权限系统。要对不认识的公众开放注册，还需要账号、配额、限流和独立数据隔离。
