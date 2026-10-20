// Pure layout calculation: no device identifiers, network, or permissions.
export function wechatNavigationLayout(windowInfo = {}, capsule = {}) {
  const width = Number(windowInfo.windowWidth) > 0 ? Number(windowInfo.windowWidth) : 375;
  const statusValue = Number(windowInfo.statusBarHeight);
  const statusBar = Math.max(0, Math.min(80, Number.isFinite(statusValue) ? statusValue : 24));
  const top = Number(capsule.top), height = Number(capsule.height), left = Number(capsule.left);
  const valid = Number.isFinite(top) && top >= statusBar && top <= statusBar + 40
    && height >= 20 && height <= 48 && left > width / 2 && left < width;
  const navHeight = valid ? Math.max(44, 2 * (top - statusBar) + height) : 48;
  // The page already has 22px horizontal padding; preserve a gap to the capsule.
  const rightSpace = valid ? Math.max(85, width - left - 22 + 12) : 96;
  return { header: { paddingTop: `${statusBar}px` }, row: { minHeight: `${navHeight}px`, paddingRight: `${rightSpace}px` } };
}
