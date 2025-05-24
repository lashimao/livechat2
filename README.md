# Amadeus System New Alpha

一个全新的实验版本, EL PSY CONGROO~

注意，此版本已经经过重构，和初版已经不同，文档已经更新，请查看文档。

## 🤝 参与贡献

欢迎加入 Amadeus System 的开发！我们期待你的贡献：

- 🌟 提交 Issue 报告 Bug 或提出新功能建议
- 📝 改进文档内容
- 🔧 修复已知问题
- ✨ 开发新功能
- 🎨 改进用户界面



如果你想在自己的服务器上部署，可以使用 Docker Compose 进行部署。

#### 准备工作

1. 确保你的服务器已安装 [Docker](https://docs.docker.com/get-docker/) 和 [Docker Compose](https://docs.docker.com/compose/install/)
2. 准备好所有必需的环境变量（参考上方环境变量配置说明）

#### Docker Compose 配置

创建 `docker-compose.yml` 文件，内容如下：

```yaml
version: '3'
services:
  container:
    image: ghcr.io/ai-poet/amadeus-system-new-alpha
    ports:
      - "3002:3002"  # 服务端口
    environment:
      - VITE_APP_DEFAULT_USERNAME=${VITE_APP_DEFAULT_USERNAME}
      - WEBRTC_API_URL=${WEBRTC_API_URL}
    restart: unless-stopped
    networks:
      - amadeus-network
    volumes:
      - ./logs:/app/service/logs  # 日志持久化存储
networks:
  amadeus-network:
    driver: bridge
```

#### 部署步骤

1. 创建 `.env` 文件，填入所需的环境变量
2. 在 `docker-compose.yml` 所在目录运行：
```bash
docker-compose up -d
```
3. 服务将在后台启动，可以通过以下命令查看日志：
```bash
docker-compose logs -f
```

### 自行部署WebRTC服务

在Zeabur模板中提供了公共WebRTC服务，但公共服务可能会不稳定，建议单独自行私有化部署WebRTC服务。

#### Docker方式部署WebRTC

克隆仓库后，进入代码仓库的service/webrtc文件夹，使用Dockerfile构建WebRTC服务镜像：

```bash
cd service/webrtc
docker build -t amadeus-webrtc-service .
```

运行WebRTC服务容器：

```bash
docker run -d --name amadeus-webrtc \
  -p 80:80 -p 443:443 -p 3478:3478 -p 5349:5349 -p 49152-65535:49152-65535/udp \
  -e LLM_API_KEY=你的AI_API密钥 \

  -e AI_MODEL=你的大语言模型名称 \
  -e MEM0_API_KEY=你的MEM0记忆服务API密钥 \
  -e TIME_LIMIT=你的WebRTC流的最大时间限制(秒) \
  -e CONCURRENCY_LIMIT=你的最大并发连接数 \
  amadeus-webrtc-service
```

#### WebRTC服务环境变量说明

以下是WebRTC服务的内置AI服务的环境变量说明，可以用于搭建公共服务：

#### 端口配置要求

部署WebRTC服务时，需要确保服务器以下端口已开放：

- 80: HTTP通信
- 443: HTTPS通信
- 3478: STUN/TURN服务（TCP）
- 5349: STUN/TURN服务（TLS）
- 49152-65535: 媒体流端口范围（UDP）

> **注意**
> 
> 如果使用云服务提供商（如AWS、阿里云等），请确保在安全组/防火墙设置中开放这些端口。

#### TURN服务器部署

在生产环境中，为了处理复杂网络环境下的音视频穿透问题，通常需要部署TURN服务器。你可以：

- 自行部署Coturn
- 参考FastRTC部署文档进行AWS自动化部署

##### 使用AWS自动部署TURN服务器

FastRTC提供了一个自动化脚本，可在AWS上部署TURN服务器：

1. 克隆FastRTC部署仓库
2. 配置AWS CLI并创建EC2密钥对
3. 修改参数文件，填入TURN用户名和密码
4. 运行CloudFormation脚本自动部署

详细步骤请参考FastRTC的自托管部署指南。

部署完成后，可在WebRTC服务的代码中填入TURN服务器信息：

```json
{
  "iceServers": [
    {
      "urls": "turn:你的TURN服务器IP:3478",
      "username": "你设置的用户名",
      "credential": "你设置的密码"
    }
  ]
}
```

> **提示**
> 
> 正确配置TURN服务器后，即使在复杂的网络环境（如对称NAT、企业防火墙后）也能保证音视频通信的稳定性。
