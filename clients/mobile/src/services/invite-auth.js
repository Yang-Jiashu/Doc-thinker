import { activateInviteSession, clearInviteSession, getPreferences, inviteOrigin, selectInviteMode } from './api.js';

let pending = null;
let authGeneration = 0;
export const INVITE_PAGE = '/pages/invite/index';
export function getInviteIdentity() { return getPreferences().inviteUser || null; }
export function enterInvitation() { return selectInviteMode(); }
export function cancelInviteRequest() { authGeneration += 1; pending?.cancel(); }
function safeError(message) { const error = new Error(message); error.inviteSafe = true; return error; }

function authRequest(path, {method = 'GET', data, token} = {}) {
  if (pending) return Promise.reject(new Error('正在处理，请稍候'));
  const origin = inviteOrigin();
  return new Promise((resolve,reject) => {
    const controller = new AbortController();
    const operation = {done:false};
    const finish = (error,value) => {
      if (operation.done) return;
      operation.done = true; clearTimeout(timer);
      if (pending === operation) pending = null;
      error ? reject(error) : resolve(value);
    };
    operation.cancel = () => { finish(new Error('操作已取消，请重新操作')); controller.abort(); };
    pending = operation;
    const timer = setTimeout(() => { finish(new Error('连接超时，没有自动重试，请稍后再试')); controller.abort(); },10000);
    const headers = {'Content-Type':'application/json'};
    if (token) headers.Authorization = 'Bearer ' + token;
    // No URL credentials, cookies, redirect following, persistence or raw server error text.
    Promise.resolve().then(() => {
      if (operation.done) throw safeError('操作已取消，请重新操作');
      return fetch(origin+'/api/v1/invite/'+path,{
      method,headers,body:data === undefined ? undefined : JSON.stringify(data),
      credentials:'omit',redirect:'error',cache:'no-store',signal:controller.signal,
      });
    }).then(async response => {
      if (!response.ok) {
        const labels = {400:'输入不符合要求，请核对后重试',401:'登录未通过，请核对受邀用户名、密码或邀请码',403:'此入口暂不可用，请联系邀请人',409:'无法完成注册，请核对邀请信息或改用登录',422:'输入不符合要求，请核对后重试',429:'尝试过于频繁，请稍后再试',503:'服务尚未准备好，请联系邀请人'};
        throw safeError(labels[response.status] || '服务暂时无法完成操作，请稍后手动重试');
      }
      return response.json();
    }).then(value => finish(null,value),error => finish(error?.inviteSafe ? error : new Error('无法读取邀请服务响应，请检查网络后手动重试')));
  });
}
export async function checkInvitationService() {
  const result = await authRequest('status');
  if (result?.service !== 'docthinker-invite' || result?.registration !== 'invitation-only') throw new Error('当前网站不是兼容的邀请入口，请联系邀请人');
  return true;
}
export function validateInvitationForm({username,password,invite_code},register = false) {
  if (typeof username !== 'string' || !/^[A-Za-z0-9_-]{3,32}$/.test(username)) return '请输入邀请人为你预留的用户名：3–32 位字母、数字、下划线或短横线';
  const count = Array.from(password || '').length;
  if (typeof password !== 'string' || count < 12 || count > 128 || !password.trim() || new TextEncoder().encode(password).length > 512) return '密码需为 12–128 个字符，不能全为空格';
  if (register && (typeof invite_code !== 'string' || !/^[A-Za-z0-9_-]{32,128}$/.test(invite_code))) return '请输入邀请人提供的完整邀请码（32–128 位），不要填网址';
  return '';
}
export async function authenticateInvitation(fields, {register = false} = {}) {
  if (getInviteIdentity()) throw new Error('请先退出当前账号，再登录其他账号');
  const issue = validateInvitationForm(fields,register);
  if (issue) throw new Error(issue);
  selectInviteMode();
  const generation = authGeneration;
  const data = {username:fields.username,password:fields.password};
  if (register) data.invite_code = fields.invite_code;
  const result = await authRequest(register ? 'register' : 'login',{method:'POST',data});
  if (generation !== authGeneration) throw new Error('操作已取消，请重新登录');
  return activateInviteSession(result).inviteUser;
}
export async function logoutInvitation({leave = false} = {}) {
  const token = getPreferences().connectionKind === 'invite' ? getPreferences().accessToken : '';
  cancelInviteRequest(); clearInviteSession({leave});
  if (token) {
    try { await authRequest('logout',{method:'POST',token}); }
    catch { /* Local credentials remain cleared even if revocation cannot reach the server. */ }
  }
}
export function reloadInvitation({leave = false} = {}) {
  // Deliberate reload destroys old user views, drafts and in-memory voice buffers.
  const origin = inviteOrigin();
  window.location.replace(origin + window.location.pathname + (leave ? '#/pages/chat/index' : '#' + INVITE_PAGE));
  window.location.reload();
}
