# 移动端语音接口

这是可选的「录音转文字 → 现有 RAG 问答 → 用户点击朗读」能力，不是实时全双工语音 Agent。不做声纹识别、声音克隆、说话人画像，也不自动录音或自动朗读。

语音服务默认关闭，与对话模型使用独立配置。仅配置问答模型的 API Key **不会**启用语音，也不会把该密钥转发给语音厂商。

## 开启方式

由独立的移动端应用挂载接口；还须按移动端部署说明开启移动访问并配置访问令牌。手机只持有移动访问令牌，不应持有 ASR/TTS 厂商密钥。

服务读取以下环境变量，不主动读取或修改 `.env`，不自动选择厂商、下载模型或探测付费接口：

| 变量 | 用途 |
| --- | --- |
| `DOCTHINKER_VOICE_ENABLED` | 默认关闭，明确设为 `true` 才允许调用 |
| `VOICE_ASR_BASE_URL` | 转写服务的 OpenAI 兼容基础地址 |
| `VOICE_ASR_KEY` | 仅供转写使用的密钥 |
| `VOICE_ASR_MODEL` | 转写模型 ID，必须明确配置 |
| `VOICE_TTS_BASE_URL` | 朗读服务的 OpenAI 兼容基础地址 |
| `VOICE_TTS_KEY` | 仅供朗读使用的密钥 |
| `VOICE_TTS_MODEL` | 朗读模型 ID，必须明确配置 |
| `VOICE_TTS_VOICE` | 服务支持的普通音色名称，必须明确配置 |

ASR 和 TTS 可单独启用，也可接不同服务。基础地址例如 `https://你的语音网关/v1`，不要填完整的 `/audio/transcriptions` 或 `/audio/speech` 路径。不会跟随重定向，也不读取 HTTP 代理环境变量。非本机回环地址必须使用 HTTPS 并提供该角色的专用密钥，否则该角色保持不可用；仅 `localhost` 和回环 IP 允许本地 HTTP。

以下仅演示**已经运行的本地兼容服务**的配置形状；这些模型名是占位符，不代表项目提供了模型：

```sh
export DOCTHINKER_VOICE_ENABLED=true
export VOICE_ASR_BASE_URL=http://127.0.0.1:8101/v1
export VOICE_ASR_MODEL=你已部署的转写模型ID
export VOICE_TTS_BASE_URL=http://127.0.0.1:8002/v1
export VOICE_TTS_MODEL=你已部署的朗读模型ID
export VOICE_TTS_VOICE=该服务支持的普通音色
```

仅 `localhost` 或回环 IP 的显式本地地址允许密钥为空。远程服务需要独立密钥；原生接口如果不兼容下面的协议，需要另写适配器，不能只替换一个 URL。

### 一组真实型号示例：OpenAI

如果你已有可调用音频接口的 OpenAI API 账号，官方当前列出的组合可以是 `gpt-4o-mini-transcribe` 转写、`gpt-4o-mini-tts` 朗读及普通音色 `alloy`。这只是**需要你主动配置和付费额度的示例**，不是本项目默认值，也不表示你的账号已获得访问权限。[转写模型](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create)、[朗读模型与音色](https://developers.openai.com/api/reference/python/resources/audio/subresources/speech/methods/create)

```sh
export DOCTHINKER_VOICE_ENABLED=true
export VOICE_ASR_BASE_URL=https://api.openai.com/v1
export VOICE_ASR_MODEL=gpt-4o-mini-transcribe
export VOICE_ASR_KEY=替换为你主动配置的语音API密钥
export VOICE_TTS_BASE_URL=https://api.openai.com/v1
export VOICE_TTS_MODEL=gpt-4o-mini-tts
export VOICE_TTS_VOICE=alloy
export VOICE_TTS_KEY=替换为你主动配置的语音API密钥
```

密钥只配置在服务端；不要贴入手机页面、仓库或截图。此示例未执行真实模型调用。

## 实际协议与客户端使用

统一入口为 `/api/v1/mobile/voice`，所有请求携带移动访问令牌 `Authorization: Bearer …`。

- `GET /capabilities`：返回 `enabled`、`asr.available`、`tts.available`、`max_audio_bytes`、`max_text_chars` 和 `accepted_formats`。不会返回厂商地址、模型 ID 或密钥；“available”只表示配置完整，不证明上游已在线或账号已开通。
- `POST /transcribe`：以 `multipart/form-data` 上传一个名为 `file` 的文件；返回 `{"text":"识别文字"}`。不自动发送到问答、记忆或知识库。客户端应让用户检查或修改文字后再提交到当前会话。
- `POST /speak`：JSON `{"text":"要朗读的文字"}`；返回 MP3 字节，`Content-Type: audio/mpeg`。由用户点击触发；客户端负责播放、停止及清理临时播放文件。

转写适配器向上游发送 `POST /audio/transcriptions`，字段为文件 `file`、`model` 和 `response_format=json`，读取返回对象中的 `text`；这是官方转写接口的请求形状。[官方转写协议](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create)

朗读适配器发送 `POST /audio/speech`，JSON 字段为 `model`、`voice`、`input`、`response_format=mp3`。目前只提供普通音色名称，不暴露自定义声音对象、声音训练或克隆接口。[官方语音合成协议](https://developers.openai.com/api/reference/python/resources/audio/subresources/speech/methods/create)

## 格式、成本和隐私边界

- 单个录音最多 **10 MiB**；总 multipart 请求额外允许 64 KiB 表单封装，最多一个文件且不接受额外表单字段。文本朗读最多 **1500 个字符**，JSON 请求体最多 16 KiB。超限明确拒绝，不静默截断。
- 检查 MP3、WAV、M4A、AAC、WebM 的容器头及 MIME 对应关系。移动客户端使用 `application/octet-stream` 时仍以文件头判定类型；文件名不决定格式，也不会原样转发给厂商。
- **项目接受格式不等于每家上游都支持。** 官方 OpenAI 转写列表包含 MP3、WAV、M4A、WebM 等，但未列裸 AAC；AAC 应仅用于确认支持它的兼容上游。需要时请在录音端选 MP3/WAV，当前后端不自动转码。[官方支持格式](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create)
- 头部检查不是完整音频解码，不能保证音轨存在、容器完整或音频时长。本接口只做短录音交互；上传大小不能替代音频时长或费用预算，部署方还需按厂商限制和账号额度控制使用。
- 上传读取期限 15 秒；上游请求总期限 30 秒，并设置连接/读取超时。每进程最多 2 个并行上游请求、4 个正在处理的写请求；繁忙直接拒绝，不排队重试。多进程会各自计数，不能当作跨实例全局限流。HTTP 取消是协作式取消，不保证停止上游已经计费的计算。
- 上游转写 JSON 最多 256 KiB、转录文本最多 20,000 字符；朗读返回最多 10 MiB，必须匹配 MP3 头与允许的 MIME。不会把任意 HTML/JSON 错误内容当音频返回。
- 服务不建立音频文件库、不缓存音频响应、不记录录音、朗读文本或转录日志。`UploadFile` 的 multipart 解析可能使用框架临时文件，会在请求结束时关闭清理；这不是“数据从未碰过临时磁盘”的保证。部署日志、反向代理和所选云厂商的保留政策需另外检查。
- 密钥留在后端。厂商错误、验证错误均返回固定消息，不返回厂商原始响应或输入值。返回设置 `Cache-Control: no-store`，不把转录结果自动写入记忆。

当前接入仅适合带访问令牌的单拥有者开发实例；不是多租户鉴权、正式微信登录或完整生产语音网关。对外部署还需要 HTTPS、入口请求限流、账号隔离、隐私告知及客户端权限说明。合成音频应向用户明确标为 AI 生成。[官方语音使用说明](https://developers.openai.com/api/docs/guides/text-to-speech)

## App 端文件与生命周期

uni-app **JS App** 的朗读分支使用 HTML5+ `plus.downloader.createDownload`：一次带访问码的 POST 直接把 `/mobile/voice/speak` 的 MP3 保存到 `_doc/`，不先发另一遍合成请求，也不依赖小程序的 `uni.getFileSystemManager`。基座缺少下载、清理或播放器能力时，在发起网络请求前报错。H5 与微信小程序仍使用各自原来的播放流程。[HTML5+ 下载接口](https://www.html5plus.org/doc/zh_cn/downloader.html)

- 原生下载关闭重试；响应等待期限 45 秒，另设 45 秒 JS 总时限并在完成时复核。只有 HTTP 200、MP3 MIME、非空且不超过 10 MiB 的下载才播放；进度事件发现超限即中止，但这不是操作系统层逐字节硬限制。
- 取消、失败、超时会中止本次下载；完成后也移除该任务记录，避免留下包含请求文本及访问码的历史下载任务。不清理其他任务，也不恢复历史任务。正常结束、取消和播放异常后，使用 `plus.io.resolveLocalFileSystemURL` 与文件 `remove` 清理本次音频；取消后迟到的文件也会丢弃。[HTML5+ 文件接口](https://www.html5plus.org/doc/zh_cn/io.html)
- 录音自动停止后只提示“待转文字”，仍需用户点击；录音错误会通知页面复位。支持系统中断事件的微信运行时会停止本次录音，不在中断结束后自动恢复。[录音事件与兼容范围](https://uniapp.dcloud.net.cn/api/media/record-manager.html)

上述 App 分支仅经过离线模拟测试，**没有真机录音、合成、播放验证**。JS 被系统暂停或 App 被强杀时，不能保证定时器及时执行、文件与原生任务记录完成清理；也不能保证取消会停止服务端已开始的计算或费用。不要把它当作具备强制安全擦除或零留存保证的语音客户端。

## 离线验证

`tests/test_voice_api_unit.py` 使用内存录音和 `httpx.MockTransport` 覆盖关闭状态、独立配置、读取前鉴权、分块上传大小、MIME/文件头、转写/朗读成功、原始错误脱敏、超时、容量和取消释放。不调用真实语音接口，不下载模型，也不读取开发者的 `.env`。这些测试不证明某个真实厂商、音色、麦克风或小程序播放接口已经连通。

接口资料核查日期：2026-09-19。
