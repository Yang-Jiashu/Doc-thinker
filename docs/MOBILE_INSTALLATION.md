# 安卓与苹果安装方式

核查日期：2026-09-19。当前实现提供 **H5 网页 App（PWA）**：可从支持的手机浏览器添加到主屏幕，联网准备后可离线体验内置演示。它不是本仓库已经签名生成的 APK / IPA；原生包仍需开发工具、应用标识及签名材料。

## 先明确目前能做什么

| 路径 | 当前产物 | 还需要什么 |
| --- | --- | --- |
| 安卓 Chrome / 苹果 Safari 主屏幕网页 App | `clients/mobile/dist/build/h5/`，含安装清单和离线静态界面 | 手机能访问的可信 HTTPS 地址；用户手动确认安装 |
| 安卓原生 App | `npm run build:app` 生成的 uni-app 资源 | HBuilderX/打包环境、DCloud AppID、包名及 Android 签名配置；再生成并验证 APK |
| 苹果原生 App | 同一套 uni-app App 资源 | HBuilderX/相匹配的原生 SDK 与 Xcode（本地打包时）、Bundle ID、适用于分发方式的证书与描述文件；再生成并验证 IPA |

PWA 的“安装”由浏览器提供，不要求项目自己创建可执行安装包；安卓浏览器/设备可能得到独立网页 App，也可能仅创建快捷方式，不能保证所有设备体验一致。[Google PWA 安装说明](https://web.dev/learn/pwa/installation)

## 电脑本地预览

在 `clients/mobile` 执行：

```sh
npm ci --ignore-scripts
npm run build:h5
npm run preview:h5
```

打开 `http://127.0.0.1:5175`。设置页有“添加到手机桌面”说明；生产构建会准备离线界面。开发热更新仍使用 `npm run dev:h5` 的 **5174**，与生产 PWA 预览分开，防止旧 service worker 干扰开发代码。

**这里的 localhost / 127.0.0.1 只代表当前电脑，不是给手机的安装地址。** 预览命令只监听本机，没有公开到公网。手机访问电脑的普通局域网 HTTP 地址不能获得同等安全环境保证；正式手机使用需要可信 HTTPS。浏览器通常为本机回环地址提供开发例外，不能把这个例外套用到任意局域网 IP。[Google 本地 HTTPS 说明](https://web.dev/articles/when-to-use-local-https)

当前没有部署 HTTPS 手机地址，不能仅凭本地构建成功就声称已经能在你的手机上安装。

## 手机添加到主屏幕

先把完整 H5 构建目录部署到你控制的 HTTPS 静态站点，保持构建文件原样；本任务不自动部署、注册域名、开启公网隧道或登录账号。当前 H5 默认根路径部署和 hash 路由；不要只上传 `index.html`，也不要把 API 路由重写成它。跨域私人服务需要配置相应 H5 来源的 CORS；HTTPS 页面连接的私人服务也应使用 HTTPS。

### 安卓

1. 用 Chrome 打开部署后的网址。
2. 在设置页出现安装按钮时点它；也可以打开浏览器菜单，选择“安装应用 / 安装并创建快捷方式 / 添加到主屏幕”（名称随版本变化）。
3. 按系统提示完成，再从桌面图标打开。

没有出现页面内按钮不等于已经安装，浏览器决定是否提供安装提示。不要为此关闭手机安全检查。[Chrome Android 官方安装步骤](https://support.google.com/chrome/answer/9658361?co=GENIE.Platform%3DAndroid&hl=en)

### 苹果

1. 在 Safari 打开网址；微信内置浏览器里请先转到 Safari。
2. 点击分享菜单，选择“添加到主屏幕”。
3. 系统如显示“作为网页 App 打开”，将它开启，然后点“添加”。较旧版本的菜单可能不同。

iPhone 不依赖 Chrome 的 `beforeinstallprompt` 按钮；按 Safari 菜单手动添加即可。这里不需要购买 Apple 开发者账号。[Apple 官方主屏幕网页 App 步骤](https://support.apple.com/guide/iphone/open-as-web-app-iphea86e5236/ios)

## 离线范围、隐私与升级

- 首次必须联网完成静态界面准备，再关闭页面并重开。缓存可能被浏览器或系统回收，因此不是永久离线存储承诺。
- 离线可使用内置演示问答、示例资料和演示图谱；这些仍明确标为演示，不是真实模型回答。
- **真实问答、资料上传、图谱接口、鉴权和语音需要联网。** Service worker 不拦截这些请求，不把失败替换成“成功”，不排队补发，不缓存私人接口的响应或录音。
- 缓存白名单来自本次构建的 `index.html`、JS/CSS/字体、品牌图片及安装配置；预缓存请求不带 Cookie/Authorization，校验构建内容摘要。未知 URL、外部域名、带查询参数/鉴权头的请求不走此离线缓存。
- 升级仅清理本应用同一 scope 下旧的静态缓存，不清空其他站点缓存、设置、草稿或数据库。新版本等待旧窗口关闭，**不会自动刷新正在编辑的页面**；看到更新提示时先保存/复制草稿，再关闭所有窗口并重开。主动关页本身仍可能丢失尚未保存的内存草稿。[Service worker 官方生命周期说明](https://web.dev/articles/service-worker-lifecycle)
- 删除桌面图标不一定清空网站存储；需要清理时使用浏览器的网站数据管理功能，不要把“卸载图标”等同于安全擦除。[Google 安装与存储说明](https://web.dev/learn/pwa/installation)

## 原生 APK / IPA 的下一步

`npm run build:app` 是**资源编译**，不是签名安装包。不要把 `dist/build/app` 目录重命名为 APK 或 IPA。

原生 `manifest.json` 已按 [uni-app 图标配置](https://uniapp.dcloud.net.cn/collocation/manifest-app) 连接 Android 五档密度和 iPhone/iPad/App Store 所需图标；可编辑 SVG 与 PNG 位于 `src/static/brand`。图标不透明且未预裁圆角，这只是准备好资源，不代表已通过商店审核。现有工具链依赖告警和客户身份隔离仍是上线前阻碍，详见[开发预览边界](MOBILE_APP.md)。

1. 由应用拥有者安装 HBuilderX 并准备自己的 DCloud AppID（它与微信 AppID、Apple Bundle ID 是不同体系），确认应用包名、权限文案和品牌资源。[DCloud 应用标识说明](https://uniapp.dcloud.net.cn/dev/app/introduction.html)
2. Android：选择 HBuilderX 支持的打包流程；本地离线方式还需要 Android Studio/SDK、对应 App 离线 SDK 和 AppKey。使用自己的发布签名配置，生成 APK 后在目标手机测试安装、权限、文件选择、录音和播放。[DCloud Android 离线环境](https://nativesupport.dcloud.net.cn/AppDocs/usesdk/android)
3. iOS：本地离线方式需要官方要求版本的 Xcode、匹配 HBuilderX 的离线 SDK，配置自己的 Bundle ID、签名身份与描述文件；具体允许哪些设备安装取决于开发、Ad Hoc、TestFlight 或 App Store 等分发方式，不能把 IPA 随意发给所有 iPhone 安装。[DCloud iOS 环境](https://nativesupport.dcloud.net.cn/AppDocs/usesdk/ios)、[Apple 开发者分发资格](https://developer.apple.com/programs/enroll/)
4. 云端打包会涉及第三方平台、账号和可能的代码/签名材料传输，必须由拥有者明确选择；本轮没有登录、创建账号、生成/读取证书、上传代码签名材料或付费操作。真机基座与正式打包是不同流程，构建成功也不能代替真机验证。[DCloud 真机运行说明](https://uniapp.dcloud.net.cn/tutorial/run/run-app.html)

## 本次环境核验边界

只读检查了当前 Mac 的 `/Applications`、用户 Applications、常见 Android SDK 路径和 `xcode-select -p`。常见位置未发现 HBuilderX、Android Studio/SDK 或完整 Xcode；`xcode-select` 指向 `/Library/Developer/CommandLineTools`。系统自带的 `xcodebuild` 命令占位符不代表已安装完整 Xcode。

没有读取 `.env`、钥匙串、证书、签名配置或开发者账号。也没有检查所有磁盘路径，因此不能断言用户完全没有其他位置的工具。本次交付不包含已签名 APK / IPA，也没有完成安卓或苹果真机安装验证。
