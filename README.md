# 汐乐曲库分析与分析台容器

一个容器 `xiyue-taste` 同时运行曲库分析和网页分析台：只读扫描音乐目录，将音频特征、标签、歌词语种和每首歌最相近的 20 首写入 `xiyue-taste-v1.json.gz`，在局域网提供分析状态、曲库画像、曲目详情及 App 结果下载。曲库分析不调用外部 API，不修改音乐文件或标签。

## 在飞牛 NAS 部署

1. 在飞牛安装并启动 Docker，将整个 `taste-analyzer` 目录放到 NAS，例如 `/vol1/1000/docker/taste-analyzer`。
2. 打开飞牛终端或 SSH，进入该目录。把下面的音乐目录和输出目录改成自己的真实绝对路径：

   ```bash
   cd /vol1/1000/docker/taste-analyzer
   export MUSIC_DIR='/vol1/1000/Music'
   export OUT_DIR='/vol1/1000/xiyue-taste'
   mkdir -p data "$OUT_DIR"
   docker compose up -d --build
   ```

3. 在同一个终端查看运行情况：

   ```bash
   docker compose logs -f taste
   ```

音乐目录挂载到 `/music:ro`；缓存保存在本目录的 `data/cache.sqlite`；输出保存在 `$OUT_DIR/xiyue-taste-v1.json.gz`。容器使用 2 个分析进程，限制为 2 CPU、2 GB 内存。启动后扫描一次，随后每隔 24 小时再扫描。

重新打开终端执行 Compose 命令时，需要先重新设置上述 `MUSIC_DIR` 和 `OUT_DIR`。NAS 上的 Docker 需要有音乐目录的读取权限，以及缓存目录、输出目录的写入权限。

## 手动扫描一次

在分析台点「立即扫描」。扫描期间再次提交请求，会在本轮结束后立即触发下一轮。

Python 命令行也可以独立运行：

```bash
python -m analyzer scan --music /music --data /data --out /out --workers 2
python -m analyzer loop --music /music --data /data --out /out --workers 2 --interval-hours 24
python -m analyzer run --music /music --data /data --out /out --workers 2 --interval-hours 24 --port 8790
```

`--workers` 默认 2；`loop` 和 `run` 的 `--interval-hours` 默认 24。`run` 在同一个服务进程里提供 HTTP，并使用后台线程处理请求；扫描工作使用独立分析进程。

## 分析台和结果下载

现在只有一个容器 `xiyue-taste`，在 8790 端口同时提供分析台和结果下载，只绑定 IPv4。输出目录可写，扫描结束会更新结果文件。

浏览器打开 `http://<NAS 局域网 IP>:8790/` 是分析台；`http://<NAS 局域网 IP>:8790/xiyue-taste-v1.json.gz` 是汐乐 App 使用的文件。

文件响应保留 `Content-Encoding: gzip`、`Content-Type: application/json` 和 `Last-Modified`，支持 `If-Modified-Since`，文件没变时返回 304。没有登录验证，任何能打开页面的人都能点「立即扫描」，不要把这个端口映射到公网。

## 说一句找歌

`POST /api/ask` 接收 JSON `{"q":"下雨天安静一点的"}`，去掉首尾空白后限 1～100 字。返回 `reason` 和最多 20 个 `items`，每项包含分析结果里的 `index`、`title`、`artists`、`path`。

Compose 从同目录的 `.env` 读取以下环境变量并传给容器：

- `TASTE_LLM_API_KEY`：硅基流动密钥，只放在 `.env`，不要写入代码、日志或提交到 Git；不填时接口返回 503。
- `TASTE_LLM_BASE_URL`：默认 `https://api.siliconflow.cn/v1`。
- `TASTE_LLM_MODEL`：默认 `deepseek-ai/DeepSeek-V3.2`。

使用此接口会把查询原文、曲库歌名、歌手、风格、语种、BPM 和响度发给模型服务商。

## 从旧版本升级

在设置好 `MUSIC_DIR` 和 `OUT_DIR` 的终端中执行：

```bash
docker compose down
docker compose up -d --build
```

第一条命令使用旧版本的 Compose 文件执行，清掉旧的 `analyzer`、`web` 两个容器，再换成新版文件启动。`data/` 和输出目录保持不动，缓存还在，未变化的歌曲不会重新分析。

## 数据与增量规则

- 支持扩展名 `flac mp3 m4a aac wav aiff ape ogg opus wma dsf`，不区分大小写，跳过隐藏目录。
- 路径相对于音乐根目录。路径、大小、修改时间和分析器版本全部不变时，不再提取特征；已删除文件从缓存移除。
- 单个文件分析失败时，只在缓存 `error` 列记录异常类名，不重试，也不进入输出。文件或分析器版本改变后才重新分析。
- 每首歌取从总时长 30% 处开始的最多 60 秒；不足 60 秒时读取全曲。提取 53 维特征，再标准化、分组加权和 L2 归一化。
- 标签缺失为 `null`，标题缺失使用不含扩展名的文件名。歌词优先使用同名 `.lrc`，其次使用内嵌歌词；没有歌词时语种为 `null`。
- 输出 schema 为 1，时间为 UTC。向量和相似度保留 4 位小数，近邻用输出 `tracks` 数组下标表示，排除自身。
- 输出先写临时 gzip 文件，再原子替换正式文件。

## 本地验证

```bash
docker build -t xiyue-taste .
docker run --rm -v "$PWD/tests:/app/tests:ro" xiyue-taste python -m pytest -q -p no:cacheprovider tests
```

解码依赖 ffmpeg，测试在镜像里跑。

测试仅生成合成音频，不使用真实歌曲。

读取输出曲目数：

```bash
python3 -c "import gzip,json;d=json.load(gzip.open('out/xiyue-taste-v1.json.gz'));print(len(d['tracks']))"
```
