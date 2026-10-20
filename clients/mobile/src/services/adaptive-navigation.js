// Only the existing five tab pages are allowed. Navigation never changes a
// connection, session, draft, or model setting, and does not read application data.
export const DESKTOP_NAVIGATION_WIDTH = 208;
export const DESKTOP_NAVIGATION_MIN_WIDTH = 1100;
export const desktopNavigationItems = Object.freeze([
  { id: 'chat', label: '助手', detail: '讨论与写作' },
  { id: 'documents', label: '资料', detail: '原文与出处' },
  { id: 'memory', label: '记忆', detail: '你确认的信息' },
  { id: 'graph', label: '图谱', detail: '查看知识联系' },
  { id: 'settings', label: '设置', detail: '连接与偏好' },
].map(item => Object.freeze({
  ...item,
  path: `/pages/${item.id}/index`,
  icon: `/static/brand/tab-${item.id}.png`,
  activeIcon: `/static/brand/tab-${item.id}-active.png`,
})));

export function desktopNavigationRoute(value) {
  if (typeof value !== 'string' || value.length > 2048 || /[\u0000-\u0020\u007f\\]/u.test(value)) return '';
  // The H5 entry route is the first tab. Query parameters do not become routes.
  const route = value.replace(/^#/, '').split('?')[0];
  if (route === '/') return '/pages/chat/index';
  const normalized = route.startsWith('/') ? route : `/${route}`;
  return desktopNavigationItems.find(item => item.path === normalized)?.path || '';
}

export function desktopNavigationTarget(id) {
  return desktopNavigationItems.find(item => item.id === id)?.path || '';
}

export function switchDesktopPage(id, currentRoute, switchTab, callbacks = {}) {
  const target = desktopNavigationTarget(id);
  const current = desktopNavigationRoute(currentRoute);
  if (!target || !current || target === current || typeof switchTab !== 'function') return false;
  switchTab({ url: target, success: callbacks.success, fail: callbacks.fail, complete: callbacks.complete });
  return true;
}
