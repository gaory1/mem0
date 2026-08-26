# LOCAL — 本地自托管部署笔记

> 本文件只记**和上游默认不同的本地差异**。代码事实以 `server/main.py` 为准。

---

## 1. 架构总览

`server/docker-compose.yaml` 三个服务（`name: mem0-dev`，容器前缀 `mem0-dev-*`）：

| 服务 | 镜像/构建 | 宿主端口 → 容器 | 用途 |
|------|-----------|------------------|------|
| `mem0` | `server/dev.Dockerfile` | `8888 → 8000` | FastAPI REST server（mem0 API） |
| `postgres` | `pgvector/pgvector:pg17` | `8432 → 5432` | 向量库 + app 数据 |
| `mem0-dashboard` | `server/dashboard` | `${DASHBOARD_PORT:-3000} → 3000` | 管理 UI |

---

## 2. 配置系统（`server/main.py` → `DEFAULT_CONFIG`）

启动时由环境变量构建 `DEFAULT_CONFIG`（`main.py:127`），再 `initialize_state()` 装入运行时。

### 2.1 LLM（主模型）— provider 写死 `openai`

LLM 固定 OpenAI 兼容客户端，靠以下环境变量调：

| 变量 | 作用 | 默认 |
|------|------|------|
| `MEM0_DEFAULT_LLM_MODEL` | 模型名 | `gpt-5-mini` |
| `OPENAI_API_KEY` | LLM 的 API key | — |
| `OPENAI_BASE_URL` | LLM 网关（指向 minimax / vLLM / 其它 OpenAI 兼容服务） | `https://api.openai.com/v1` |

### 2.2 Embedder — 完全独立可配

| 变量 | 作用 | 默认 |
|------|------|------|
| `MEM0_EMBEDDER_PROVIDER` | provider（`ollama` / `openai` / `huggingface` / `lmstudio`） | `openai` |
| `MEM0_DEFAULT_EMBEDDER_MODEL` | 模型名 | `text-embedding-3-small` |
| `MEM0_EMBEDDER_API_KEY` | embedder 的 key（本地可空） | — |
| `MEM0_EMBEDDER_BASE_URL` | base URL，会写入 `{provider}_base_url` | `http://localhost:11434/v1` |
| `EMBEDDING_DIMS` | 维度（同时喂给 embedder 和 pgvector） | `1024` |

关键映射行 `server/main.py:148`：
```python
f"{EMBEDDER_PROVIDER}_base_url": EMBEDDER_BASE_URL,
```
即 `MEM0_EMBEDDER_BASE_URL` 会被转成 `ollama_base_url` / `openai_base_url` / ... 传进 provider config。

> ⚠️ 默认 `MEM0_EMBEDDER_BASE_URL=http://localhost:11434/v1` 是 openai 取向。若 `MEM0_EMBEDDER_PROVIDER=ollama` 需要修改这个BASE URL并且在容器内安装对应的依赖包。

### 2.3 ⚠️ 持久化覆盖（".env 不生效" 头号原因）

通过 dashboard 的 `/configure` 或 `POST /configure` 改过的配置，会写进 Postgres 表 `settings` 的 `config_overrides` 行。重启时 `server/server_state.py` 的 `initialize_state()` 会把它 **merge 到 `DEFAULT_CONFIG` 之上**，**盖掉 `.env`**。

排查：
```bash
docker exec mem0-dev-postgres-1 psql -U postgres -d mem0_app \
  -c "SELECT value FROM settings WHERE key='config_overrides';"
```
有内容且不想用它 → 删掉该行（`DELETE FROM settings WHERE key='config_overrides';`），让 `.env` 重新生效。

---

## 3. Embedding：bge-m3 + Ollama（1024 维）

本地实际 `.env`：
```env
MEM0_EMBEDDER_PROVIDER=openai                      # 走 Ollama 的 OpenAI 兼容端点
MEM0_EMBEDDER_BASE_URL=http://10.4.2.240:11434/v1  # 注意带 /v1
MEM0_DEFAULT_EMBEDDER_MODEL=bge-m3:latest
MEM0_EMBEDDER_API_KEY=<任意>                        # Ollama 不校验，但 CLI/客户端常要求非空
EMBEDDING_DIMS=1024
```

### 3.1 `embedding_model_dims` 必须显式配且与模型一致

- mem0 **没有任何地方**从 embedder 推导 vector_store 维度，pgvector 配置默认 1536（`mem0/configs/vector_stores/pgvector.py:9`）。
- `DEFAULT_CONFIG` 里 `vector_store.config.embedding_model_dims = EMBEDDING_DIMS`，与 embedder 同源一个变量 → 保持一致。
- 维度不匹配 → 插入时报 `shapes (0,1536) and (1024,) not aligned`。

### 3.2 pgvector 维度建表时定死

换维度（如 1024 ↔ 1536）必须先删表，重启后按新维度重建：
```sql
DROP TABLE IF EXISTS memories;
```

### 3.3 网络坑：容器连不到宿主 ollama

容器内 `localhost:11434` 是容器自己。Ollama 在宿主时：
- 设 `MEM0_EMBEDDER_BASE_URL=http://host.docker.internal:11434`（Linux 还需 compose 加 `extra_hosts: ["host.docker.internal:host-gateway"]`）；
- 或把 ollama 作为 service 塞进 compose，用 `http://ollama:11434`；
- 或像本地现在这样用可达的局域网 IP。

### 3.4 接 Ollama 的两种姿势

| 模式 | `MEM0_EMBEDDER_PROVIDER` | URL | 说明 |
|------|--------------------------|-----|------|
| 原生 ollama | `ollama` | `http://<host>:11434`（**不带** `/v1`） | 走 `client.embed()` |
| OpenAI 兼容 | `openai` | `http://<host>:11434/v1`（**带** `/v1`） | 走 `embeddings.create()`，可与 LLM 复用客户端 |

本地用 OpenAI 兼容模式（带 `/v1`）。

### 3.5 embedder base_url 必须进 config（已踩坑）

若 embedder config 不带 `{provider}_base_url`，`mem0/embeddings/openai.py` 会回退到 `OPENAI_BASE_URL`（你的 **LLM 网关**）→ 用 embedder 的 key 打 LLM 网关 → **401**。
所以 `main.py:148` 那行 `{provider}_base_url` 映射是必须的，别简化掉。

---

## 4. CLI 连本地 server

**用 node CLI**（`@mem0/cli`）。它在 `base_url != https://api.mem0.ai` 时自动启用 `OssBackend`（`cli/node/src/backend/base.ts:getBackend`），端点（`/memories`、`/search`、`/entities`）与 `server/` 路由一致。

### 4.1 安装 + 配置

```bash
npm install -g @mem0/cli
export MEM0_API_KEY=dummy                          
mem0 config set platform.base_url http://localhost:8888
mem0 config set defaults.user_id cli                # 省去每条 --user-id
```

### 4.2 用

```bash
mem0 add "我喜欢深色主题，用 vim 键位"
mem0 search "他有什么偏好"
mem0 list
mem0 get <memory-id>
```

不持久化也可每条加 `--base-url http://localhost:8888`（每个命令都有该 flag，`cli/node/src/index.ts`）。

### 4.3 注意

- **`localhost:8888` 是宿主视角**（compose 映射 `8888:8000`）。CLI 也跑在容器里时改用 `http://mem0:8000`。
- **开启鉴权后会 401**：server 认 `X-API-Key` / `Bearer JWT` / `Authorization: Token …`。

---

## 5. 运维速查

```bash
# .env 改动后必须重建容器（uvicorn --reload 只盯 .py，不盯 .env）
cd server && docker compose up -d --force-recreate mem0

# 看日志
docker logs mem0-dev-mem0-1 --tail 50

# 确认向量维度
docker exec mem0-dev-postgres-1 psql -U postgres -d mem0_app \
  -c "SELECT memory_id, vector_dims(vector) FROM memories LIMIT 5;"

# 关遥测噪音（PostHog timeout）
# 在 .env 设 MEM0_TELEMETRY=false
```

### 可选：spaCy（提升 BM25 全文检索）

未装 spaCy 时 `mem0/utils/lemmatization.py:lemmatize_for_bm25` 回退原文，全文检索仍工作但无词形还原。需要则：`server/requirements.txt` 加 `spacy>=3.7.0`（= `mem0ai[nlp]` extra），重建镜像。

## 6. 备注
mem0 容器（后端）的 DASHBOARD_URL 是CORS 白名单，即允许哪个来源的前端跨域调后端。
mem0-dashboard 容器（前端）的 DASHBOARD_URL，仅用来判断是否https，管「refresh cookie 加不加 Secure 位」。

