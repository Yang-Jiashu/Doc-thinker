// Platform authorization is separate from consent to send data to a private
// server. Call only in a user-initiated flow, after the offline demo return.
// Do not register onNeedPrivacyAuthorization here: WeChat's official privacy
// dialog remains responsible for the actual consent button and policy link.
// https://uniapp.dcloud.net.cn/tutorial/mp-weixin-user-privacy.html

function checkActive(isCurrent) {
  if (!isCurrent()) throw new Error('操作已取消或服务配置已变化，请重新操作');
}

function callWechat(method, options, message, isCurrent) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (error, result) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (error) reject(error); else resolve(result);
    };
    // A dismissed/broken platform dialog must not leave the UI busy forever.
    // A late success cannot restart the cancelled operation.
    const timer = setTimeout(() => finish(new Error('授权等待超时，已取消此操作，请重新点击')), 60000);
    try {
      checkActive(isCurrent);
      if (typeof wx === 'undefined' || typeof wx[method] !== 'function') {
        throw new Error('当前微信版本不支持所需授权，请升级微信；暂时可使用文字或演示模式');
      }
      wx[method]({
        ...options,
        success: result => {
          try { checkActive(isCurrent); finish(null, result); }
          catch (error) { finish(error); }
        },
        // Do not expose raw platform errors or automatically reopen settings.
        fail: error => finish(new Error(Number(error?.errno) === 112 || /scope is not declared/i.test(String(error?.errMsg || ''))
          ? '小程序后台尚未完整声明所需隐私权限，请联系管理员；此操作未继续'
          : message)),
      });
    } catch (error) { finish(error); }
  });
}

export async function ensureWeChatPrivacy({mode = 'demo', action = '此功能', isCurrent = () => true} = {}) {
  if (mode !== 'connected') return;
  // #ifdef MP-WEIXIN
  checkActive(isCurrent);
  if (typeof wx === 'undefined' || typeof wx.getPrivacySetting !== 'function'
      || typeof wx.requirePrivacyAuthorize !== 'function') {
    throw new Error('当前微信版本不支持隐私授权，请升级微信；暂时可使用文字或演示模式');
  }
  const settings = await callWechat('getPrivacySetting', {}, '无法读取微信隐私授权，请检查小程序隐私配置后重试', isCurrent);
  checkActive(isCurrent);
  if (settings?.needAuthorization === false) return;
  if (settings?.needAuthorization !== true) throw new Error('微信隐私授权状态不明确，已取消此操作');
  await callWechat('requirePrivacyAuthorize', {}, `${action}需要微信隐私授权；未完成授权，已取消此操作`, isCurrent);
  checkActive(isCurrent);
  // #endif
}

export async function ensureWeChatRecordingPermission({mode = 'demo', isCurrent = () => true} = {}) {
  if (mode !== 'connected') return;
  // #ifdef MP-WEIXIN
  const settings = await callWechat('getSetting', {}, '无法读取录音权限，请检查微信设置后重试', isCurrent);
  checkActive(isCurrent);
  const permission = settings?.authSetting?.['scope.record'];
  if (permission === true) return;
  if (permission === false) throw new Error('麦克风权限已关闭，请在微信右上角「…」→「设置」中允许录音，再手动开始');
  if (!settings || !settings.authSetting || typeof settings.authSetting !== 'object') {
    throw new Error('录音权限状态不明确，已取消录音');
  }
  await callWechat('authorize', {scope:'scope.record'}, '未获得麦克风权限；没有开始录音，仍可输入文字', isCurrent);
  checkActive(isCurrent);
  // #endif
}
