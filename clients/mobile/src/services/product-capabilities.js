// Product copy is derived only from the client's selected mode, never from a
// guessed server status. This module must not read credentials or call APIs.
export function productExperience(preferences = {}) {
  preferences = preferences && typeof preferences === 'object' ? preferences : {};
  if (preferences.connectionKind === 'invite') {
    const pending = preferences.invitePending === true;
    return {
      kind: 'invite', label: pending ? '先登录，再开始' : '邀请内测工作台',
      summary: '默认提供文字对话与会话历史；研究能力以登录后各页的服务提示为准。',
      status: pending ? '等待登录' : '查看页面能力提示',
      available: ['文字对话', '会话历史', '对话上下文开关'],
      boundary: '运营者开放研究模式后，资料页可上传与核对原文，记忆页可手动保存项目结论；否则这些入口保持关闭。图谱、自动学习与语音尚未开放。聊天历史不是长期记忆。',
      steps: pending ? ['使用受邀用户名注册或登录。', '登录后新建会话，再开始提问。']
        : ['新建会话，说明你想解决的问题。', '继续追问或比较观点；内容由云模型处理，请先用非敏感内容试用。'],
      primary: { label: pending ? '前往登录' : '开始对话', page: pending ? 'invite' : 'chat' },
      secondary: pending ? null : { label: '管理账号', page: 'invite' },
    };
  }
  if (preferences.mode === 'connected') {
    return {
      kind: 'private', label: '你的私人研究工作台',
      summary: '这是客户端的连接配置，不等于后端功能已经就绪。',
      status: '能力需后端确认',
      available: ['会话与提问入口', '资料与证据浏览入口', '当前会话记忆入口'],
      boundary: '实际可用性由私人后端决定。保存配置不会验证模型、资料解析、记忆或语音；共用访问码不能代替多用户隔离。',
      steps: ['保存服务地址与访问码，手动检查连接。', '新建会话，上传一份非敏感资料；解析完成后提问并检查返回依据。'],
      primary: { label: '打开助手', page: 'chat' },
      secondary: { label: '打开资料页', page: 'documents' },
    };
  }
  return {
    kind: 'demo', label: '用自己的材料，核对自己的问题',
    summary: '当前未连接真实服务；本地示例只能预览界面，不能核对你的资料。',
    status: '尚未连接',
    available: ['本地界面预览', '功能入口说明'],
    boundary: '本地示例不读取你的文件或麦克风，也不调用模型。要处理自己的资料，需连接可用服务或受邀登录，并由后端开放对应能力；点击入口不会自动连接或切换模式。',
    steps: ['在设置中保存服务地址与访问码，手动检查连接；收到邀请链接时，在该网站登录。', '研究服务开放后，新建会话，添加自己的资料，再围绕问题查看回答与依据。'],
    primary: { label: '连接真实服务', page: 'settings' },
    secondary: { label: '查看功能入口', page: 'chat' },
  };
}

const unopenedFeatures = {
  documents: {
    name: '资料库', title: '资料接入，下一步再开放',
    description: '当前邀请内测不会读取或上传你的文件，也还不能围绕自己的论文提问。',
  },
  memory: {
    name: '长期记忆', title: '历史还在，长期记忆尚未开放',
    description: '你可以查看自己的会话历史，但它不会在这个内测版本中自动成长为长期记忆或个人画像。',
  },
  graph: {
    name: '知识图谱', title: '先把对话做好，再开放图谱',
    description: '当前邀请内测没有开放知识图谱；资料接入是否开放，以资料页提示为准。不会用示例节点冒充你的知识。',
  },
};

export function featureBoundary(feature) {
  const value = Object.prototype.hasOwnProperty.call(unopenedFeatures, feature) ? unopenedFeatures[feature] : null;
  return value ? { ...value } : {
    name: '此功能', title: '这项功能还没有开放', description: '当前邀请内测仅开放文字对话和会话历史。',
  };
}

// Fixed destinations only; callers cannot turn a product action into an arbitrary
// external link or a request containing credentials.
export function openProductPage(page) {
  if (page === 'invite') return uni.reLaunch({ url: '/pages/invite/index' });
  if (page === 'studio') return uni.navigateTo({ url: '/pages/studio/index' });
  if (page === 'chat') return uni.switchTab({ url: '/pages/chat/index' });
  if (page === 'documents') return uni.switchTab({ url: '/pages/documents/index' });
  if (page === 'settings') return uni.switchTab({ url: '/pages/settings/index' });
  return false;
}
