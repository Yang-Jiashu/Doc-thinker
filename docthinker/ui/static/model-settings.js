/* Local, versioned presets. Loading this page never calls a model provider. */
(function (root) {
  'use strict';
  const roles = {chat: '对话', vision: '视觉', keyword: '关键词', extraction: '知识抽取', embedding: '嵌入', rerank: '重排'};
  const groups = ['llm', 'embed', 'rerank'];
  const numericFields = new Set(['embed_dim', 'llm_max_async', 'embedding_max_async', 'max_parallel_insert', 'llm_router_max_concurrency']);
  const roleFields = {llm_model: 'chat', vlm_model: 'vision', keyword_llm_model: 'keyword', entity_extraction_llm_model: 'extraction', embed_model: 'embedding', rerank_model: 'rerank'};
  const endpointRole = group => group === 'embed' ? 'embedding' : group;
  const normalizeEndpoint = value => {
    try { const url = new URL(String(value).trim()); return url.origin + url.pathname.replace(/\/+$/, ''); }
    catch (_) { return String(value || '').trim().replace(/\/+$/, ''); }
  };
  function providerFor(catalog, group, endpoint) {
    const normalized = normalizeEndpoint(endpoint);
    return catalog.providers.find(provider => {
      const base = provider.endpoints[endpointRole(group)];
      if (!base) return false;
      if (normalizeEndpoint(base) === normalized) return true;
      // Workspace-specific endpoints are matched only against their exact documented suffix.
      if (!base.includes('{WorkspaceId}')) return false;
      const parts = base.toLowerCase().split('{workspaceid}');
      const actual = normalized.toLowerCase();
      const middle = actual.slice(parts[0].length, actual.length - parts[1].replace(/\/+$/, '').length);
      return actual.startsWith(parts[0]) && actual.endsWith(parts[1].replace(/\/+$/, '')) && /^[a-z0-9-]+$/.test(middle);
    });
  }
  function modelsForRole(provider, role, search = '') {
    const needle = search.trim().toLocaleLowerCase();
    return (provider?.models || []).filter(model => model.roles.includes(role) &&
      (!needle || [model.name, model.id, model.description].join(' ').toLocaleLowerCase().includes(needle)));
  }
  function buildPayload(original, values, keys) {
    const payload = {};
    for (const [field, value] of Object.entries(values)) {
      if (String(value ?? '') !== String(original[field] ?? '')) payload[field] = numericFields.has(field) ? Number(value) : value;
    }
    for (const group of groups) {
      if (keys[group]?.trim()) payload[group + '_api_key'] = keys[group].trim();
    }
    if ('llm_base_url' in payload) {
      // A base migration is a complete shared-connection selection, including
      // unchanged IDs and explicit inheritance. Match the server's safety contract.
      for (const field of ['llm_model', 'vlm_model', 'keyword_llm_model', 'entity_extraction_llm_model']) payload[field] = values[field] ?? '';
    }
    // A changed provider must not retain hidden, provider-specific extraction overrides.
    if ('llm_base_url' in payload || 'llm_model' in payload || 'entity_extraction_llm_model' in payload) payload.extraction_llm_model = '';
    return payload;
  }
  root.ModelSettings = {normalizeEndpoint, providerFor, modelsForRole, buildPayload};
  if (typeof document === 'undefined') return;
  const $ = id => document.getElementById(id);
  const fieldElement = field => $('s-' + field.replaceAll('_', '-'));
  let catalog = {providers: []}, original = {}, loaded = false, saving = false, picker = null, opener = null;
  const provider = group => catalog.providers.find(item => item.id === $('s-' + group + '-provider').value);
  const getKeys = () => Object.fromEntries(groups.map(group => [group, $('s-' + group + '-api-key').value]));
  function values() {
    const result = Object.fromEntries(Array.from(document.querySelectorAll('[data-setting]'), element => [element.dataset.setting, element.value.trim()]));
    if (!$('s-rerank-enabled').checked) {
      result.rerank_model = '';
      result.rerank_base_url = original.rerank_base_url || '';
    }
    return result;
  }
  function payload() {
    const keys = getKeys();
    if (!$('s-rerank-enabled').checked) keys.rerank = '';
    return buildPayload(original, values(), keys);
  }
  function dirty() { return loaded && (Object.keys(payload()).length > 0 || $('s-rerank-enabled').checked !== Boolean(original.rerank_model)); }
  function notice(id, message, error = false) {
    const element = $(id); element.textContent = message; element.hidden = !message;
    element.classList.toggle('ms-error', error);
  }
  function link(id, url) {
    const element = $(id);
    if (!url || !/^https:\/\//.test(url)) { element.hidden = true; element.removeAttribute('href'); return; }
    element.href = url; element.hidden = false;
  }
  function renderModels() {
    for (const [field] of Object.entries(roleFields)) {
      const group = field.startsWith('embed') ? 'embed' : field.startsWith('rerank') ? 'rerank' : 'llm';
      const id = fieldElement(field).value;
      const model = provider(group)?.models.find(item => item.id === id);
      const optional = ['vlm_model', 'keyword_llm_model', 'entity_extraction_llm_model'].includes(field);
      $('value-' + field).textContent = model?.name || (id ? id + ' · 自定义' : optional ? '跟随对话模型' : '选择模型');
      $('value-' + field).title = id;
    }
  }
  function renderGroup(group) {
    const item = provider(group);
    const base = fieldElement(group + '_base_url').value;
    const changed = normalizeEndpoint(base) !== normalizeEndpoint(original[group + '_base_url']);
    const masked = original[group + '_api_key_masked'];
    $('s-' + group + '-api-key-hint').textContent = changed ? '连接地址已变更，请重新填写此服务商的密钥。' : masked ? '已保存 ' + masked + ' · 留空保留' : '尚未配置密钥。';
    if (item?.local) $('s-' + group + '-api-key-hint').textContent += ' 本地无鉴权服务可填写 EMPTY。';
    $('s-' + group + '-api-key').placeholder = item?.local ? '本地无鉴权可填写 EMPTY' : '输入此服务商的 API Key';
    $(group + '-provider-description').textContent = item?.description || '保留你的自定义模型与连接。请确认此服务兼容项目使用的接口。';
    link(group + '-docs-link', item?.docs_url); link(group + '-key-link', item?.key_url);
    $(group + '-notes').hidden = !item?.notes?.length;
    $(group + '-notes-content').textContent = (item?.notes || []).join('\n');
    const warnings = [];
    if (/[{}]/.test(base)) warnings.unshift('此地址需要业务空间 ID。请展开高级连接，将 {WorkspaceId} 替换为你的实际值。');
    if (group === 'llm') {
      const visionID = fieldElement('vlm_model').value || fieldElement('llm_model').value.split(',')[0].trim();
      if (!item?.models.some(model => model.id === visionID && model.roles.includes('vision'))) warnings.push('当前视觉模型未在预设中确认支持图片。纯文本模型不能代替读图模型。');
      if (item && (!modelsForRole(item, 'keyword').length || !modelsForRole(item, 'extraction').length)) warnings.push('此服务商未提供独立的抽取预设；跟随对话模型时，请自行验证结构化输出与耗时。');
    }
    if (group === 'embed' && (fieldElement('embed_model').value !== original.embed_model || String(fieldElement('embed_dim').value) !== String(original.embed_dim))) warnings.push('嵌入配置已更改：已有向量不能直接混用。保存并重启后，请在新知识库中重新导入文档或执行索引重建。');
    notice(group + '-warning', warnings.join('\n'));
  }
  function refresh() {
    renderModels(); groups.forEach(renderGroup);
    $('rerank-controls').hidden = !$('s-rerank-enabled').checked;
    const changed = dirty();
    $('settings-dirty').textContent = saving ? '正在保存…' : changed ? '有未保存的更改' : original.requires_restart ? '已保存，等待后端重启' : '没有未保存的更改';
    $('save-btn').disabled = saving || !changed;
    $('reset-btn').disabled = saving || !changed;
  }
  function populateProviders() {
    for (const group of groups) {
      const select = $('s-' + group + '-provider'); select.replaceChildren();
      for (const item of catalog.providers.filter(item => item.endpoints[endpointRole(group)])) {
        const option = document.createElement('option'); option.value = item.id; option.textContent = item.name; select.append(option);
      }
      const custom = document.createElement('option'); custom.value = 'custom'; custom.textContent = '自定义 / 兼容服务'; select.append(custom);
    }
  }
  function fillForm() {
    for (const element of document.querySelectorAll('[data-setting]')) element.value = original[element.dataset.setting] ?? '';
    for (const group of groups) {
      $('s-' + group + '-provider').value = providerFor(catalog, group, original[group + '_base_url'])?.id || 'custom';
      $('s-' + group + '-api-key').value = '';
      $(group + '-advanced').open = !provider(group);
    }
    $('s-rerank-enabled').checked = Boolean(original.rerank_model);
    $('s-workdir').textContent = original.workdir || '未配置';
    notice('settings-notice', [original.requires_restart ? '配置已保存，重启后端后生效。当前运行中的客户端仍使用旧配置。' : '', ...(original.warnings || [])].filter(Boolean).join('\n'));
    refresh();
  }
  async function jsonRequest(url, options) {
    const response = await fetch(url, {...options, cache: 'no-store'});
    const data = await response.json();
    if (!response.ok) {
      // Validation details may contain submitted secrets: display only messages, never input values.
      const detail = Array.isArray(data.detail) ? data.detail.map(item => item.msg).join('；') : data.detail;
      throw new Error(typeof detail === 'string' ? detail : '请求失败，请检查后端服务。');
    }
    return data;
  }
  async function load() {
    $('settings-retry').hidden = true;
    $('settings-loading').hidden = false; $('settings-loading').textContent = '正在读取设置…';
    try {
      const [settings, presets] = await Promise.all([
        jsonRequest('/api/v1/settings'),
        jsonRequest('/static/model_catalog.json').catch(() => null)
      ]);
      catalog = presets?.providers ? presets : {providers: []};
      original = {...settings};
      if (settings.llm_models?.length > 1) original.llm_model = settings.llm_models.join(', ');
      populateProviders(); loaded = true; fillForm();
      const count = catalog.providers.reduce((sum, item) => sum + item.models.length, 0);
      $('catalog-count').textContent = catalog.providers.length + ' 服务商 · ' + count + ' 模型';
      notice('catalog-notice', presets ? '' : '模型预设暂时无法加载。现有配置不受影响，仍可使用自定义模型。');
      $('settings-form').hidden = false; $('settings-loading').hidden = true;
    } catch (error) {
      $('settings-loading').textContent = '无法读取设置：' + error.message; $('settings-retry').hidden = false;
    }
  }
  function switchProvider(group) {
    const item = provider(group);
    // Never keep a typed secret when choosing a different credential destination.
    $('s-' + group + '-api-key').value = '';
    if (item) {
      fieldElement(group + '_base_url').value = item.endpoints[endpointRole(group)];
      for (const button of document.querySelectorAll('[data-model-field][data-group="' + group + '"]')) {
        const model = modelsForRole(item, button.dataset.role)[0];
        fieldElement(button.dataset.modelField).value = model?.id || '';
        if (group === 'embed') fieldElement('embed_dim').value = model?.dimension || '';
      }
    }
    $(group + '-advanced').open = !item || /[{}]/.test(fieldElement(group + '_base_url').value);
    refresh();
  }
  function selectModel(model) {
    fieldElement(picker.field).value = model.id;
    if (picker.group === 'embed') fieldElement('embed_dim').value = model.dimension || '';
    $('model-dialog').close(); refresh();
  }
  function resultButton(model, name, description, tags = '') {
    const button = document.createElement('button'); button.type = 'button'; button.className = 'ms-model-option';
    button.setAttribute('aria-pressed', String(fieldElement(picker.field).value === model.id));
    const title = document.createElement('span'); title.className = 'ms-option-title'; title.textContent = name;
    const badges = document.createElement('span'); badges.className = 'ms-tags'; badges.textContent = tags; title.append(badges);
    const id = document.createElement('span'); id.className = 'ms-option-id'; id.textContent = model.id;
    const detail = document.createElement('span'); detail.className = 'ms-option-description'; detail.textContent = description;
    button.append(title, id, detail); button.addEventListener('click', () => selectModel(model));
    return button;
  }
  function renderResults() {
    const container = $('model-results'); container.replaceChildren();
    const search = $('model-search').value;
    if (['vision', 'keyword', 'extraction'].includes(picker.role) && !search) container.append(resultButton({id: ''}, '跟随对话模型', '不单独指定模型；视觉任务仍要求主模型具备图片理解能力。'));
    const models = modelsForRole(provider(picker.group), picker.role, search);
    for (const model of models) {
      const tags = model.roles.filter(role => ['chat', 'vision', 'embedding', 'rerank'].includes(role)).map(role => roles[role]);
      if (model.dimension) tags.push(model.dimension + ' 维');
      if (model.requires_download) tags.push('需本地下载');
      container.append(resultButton(model, model.name, model.description || '', tags.join(' · ')));
    }
    if (!models.length) {
      const empty = document.createElement('p'); empty.className = 'ms-empty'; empty.textContent = search ? '没有匹配的模型。试试其他关键词，或使用自定义模型。' : '此服务商没有该能力的预设。可使用自定义模型，并确认接口兼容。'; container.append(empty);
    }
  }
  function openPicker(button) {
    picker = {field: button.dataset.modelField, role: button.dataset.role, group: button.dataset.group}; opener = button;
    $('model-dialog-title').textContent = '选择' + roles[picker.role] + '模型';
    $('model-dialog-provider').textContent = (provider(picker.group)?.name || '自定义服务') + ' · 仅显示该角色的模型';
    $('model-search').value = ''; $('model-custom-id').value = fieldElement(picker.field).value;
    $('model-custom').open = !provider(picker.group);
    $('model-custom-hint').textContent = picker.role === 'chat' ? '模型 ID 需与服务商一致。兼容原有多模型路由，可用英文逗号分隔。' : '自定义模型需自行确认接口和能力兼容。';
    $('catalog-date').textContent = catalog.checked_at ? '预设核对于 ' + catalog.checked_at + ' · 不代表账户已开通或实时在线' : '本地预设不可用 · 自定义设置仍可使用';
    renderResults(); $('model-dialog').showModal(); $('model-search').focus();
  }
  function validate() {
    const current = values(), keys = getKeys();
    for (const group of groups) {
      if (group === 'rerank' && !$('s-rerank-enabled').checked) continue;
      const base = current[group + '_base_url'];
      let url; try { url = new URL(base); } catch (_) { throw new Error('请填写有效的 ' + group + ' 连接地址。'); }
      if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || /[{}]/.test(base)) throw new Error('请补全 ' + group + ' 连接地址，不要包含占位符或用户名密码。');
      if (normalizeEndpoint(base) !== normalizeEndpoint(original[group + '_base_url']) && !keys[group].trim()) throw new Error('连接地址已变更，请填写 ' + group + ' 服务商的 API Key；本地无鉴权服务可填 EMPTY。');
      if (!current[group + '_model']) throw new Error('请先选择' + ({llm: '对话', embed: '嵌入', rerank: '重排'}[group]) + '模型。');
    }
    for (const field of numericFields) {
      const value = current[field], limit = field === 'embed_dim' ? 65536 : 1024;
      if (!/^\d+$/.test(String(value)) || Number(value) < 1 || Number(value) > limit) throw new Error((field === 'embed_dim' ? '向量维度' : '并发参数') + '必须是 1–' + limit + ' 的整数。');
    }
  }
  async function save(event) {
    event.preventDefault(); if (saving || !dirty()) return;
    try { validate(); } catch (error) { notice('settings-notice', error.message, true); $('settings-notice').scrollIntoView({block: 'nearest'}); return; }
    const submitted = payload(); saving = true; $('settings-fields').disabled = true; refresh();
    try {
      const result = await jsonRequest('/api/v1/settings', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(submitted)});
      // Update the non-secret snapshot before a reload; a failed GET must not invite a duplicate save.
      original = {...original, ...values(), requires_restart: result.requires_restart !== false, warnings: result.warnings || []};
      for (const group of groups) if (submitted[group + '_api_key']) original[group + '_api_key_masked'] = '••••（已更新）';
      groups.forEach(group => { $('s-' + group + '-api-key').value = ''; });
      fillForm();
    } catch (error) { notice('settings-notice', '保存失败：' + error.message, true); }
    finally { saving = false; $('settings-fields').disabled = false; refresh(); }
  }
  document.addEventListener('DOMContentLoaded', () => {
    $('settings-form').addEventListener('submit', save);
    $('settings-form').addEventListener('input', refresh);
    $('s-rerank-enabled').addEventListener('change', refresh);
    for (const group of groups) {
      $('s-' + group + '-provider').addEventListener('change', () => switchProvider(group));
      fieldElement(group + '_base_url').addEventListener('input', () => {
        // A custom endpoint cannot silently retain a credential typed for a preset endpoint.
        $('s-' + group + '-api-key').value = '';
        $('s-' + group + '-provider').value = providerFor(catalog, group, fieldElement(group + '_base_url').value)?.id || 'custom'; refresh();
      });
    }
    document.querySelectorAll('[data-model-field]').forEach(button => button.addEventListener('click', () => openPicker(button)));
    $('model-search').addEventListener('input', renderResults);
    $('model-dialog-close').addEventListener('click', () => $('model-dialog').close());
    $('model-dialog').addEventListener('close', () => opener?.focus());
    $('model-custom-apply').addEventListener('click', () => {
      const id = $('model-custom-id').value.trim();
      if (!id || /[\r\n]/.test(id)) { $('model-custom-hint').textContent = '请填写非空、单行的模型 ID。'; return; }
      selectModel({id, dimension: picker.group === 'embed' ? fieldElement('embed_dim').value : undefined});
    });
    $('reset-btn').addEventListener('click', fillForm);
    $('settings-retry').addEventListener('click', load);
    root.addEventListener('beforeunload', event => { if (dirty()) { event.preventDefault(); event.returnValue = ''; } });
    load();
  });
})(typeof window === 'undefined' ? globalThis : window);
