// Screen-space annotation only: no graph positions, relations or source data are changed.
// A bounded candidate window and a uniform occupancy grid keep dense graphs legible.
export const GRAPH_LABEL_LIMITS = Object.freeze({nodes:2000,labels:45,candidates:240,cell:64,inset:8,clearance:3,viewportMax:1000000});
const ANCHORS = Object.freeze(['e','w','n','s','se','sw','ne','nw']);
const finite = value => typeof value === 'number' && Number.isFinite(value);
const compareIds = (a,b) => a < b ? -1 : a > b ? 1 : 0;
const key = (x,y) => `${x},${y}`;

let segmenter;
function graphemes(text) {
  if (typeof Intl !== 'undefined' && typeof Intl.Segmenter === 'function') {
    segmenter ||= new Intl.Segmenter(undefined,{granularity:'grapheme'});
    return Array.from(segmenter.segment(text),part=>part.segment);
  }
  // Keep surrogate pairs, combining marks, emoji modifiers and ZWJ sequences intact
  // on older clients without Intl.Segmenter. Regional indicators pair into flags.
  const result=[];
  for (const point of text) {
    const last=result[result.length-1] || '';
    const join=/[\p{Mark}\uFE00-\uFE0F\u{1F3FB}-\u{1F3FF}\u200D]/u.test(point) || last.endsWith('\u200D');
    const flag=/^[\u{1F1E6}-\u{1F1FF}]$/u.test(point) && /^[\u{1F1E6}-\u{1F1FF}]$/u.test(last);
    if (last && (join || flag)) result[result.length-1]+=point;
    else result.push(point);
  }
  return result;
}

function cleanText(value) {
  if (typeof value !== 'string') return {text:'',truncated:false};
  const bounded=value.length > 4096 ? value.slice(0,4096).replace(/[\uD800-\uDBFF]$/u,'') : value;
  const text=bounded.replace(/[\u0000-\u001F\u007F\u202A-\u202E\u2066-\u2069]/g,' ').replace(/\s+/gu,' ').trim();
  const points=Array.from(text);
  if (points.length <= 512) return {text,truncated:value.length > 4096};
  const parts=graphemes(text);
  return {text:parts.slice(0,512).join(''),truncated:value.length > 4096 || parts.length > 512};
}

function measured(text,measureText) {
  let result;
  try {result=measureText(text);} catch {return Infinity;}
  const width=typeof result === 'number' ? result : result?.width;
  return finite(width) && width >= 0 ? width : Infinity;
}

/** Returns a measured single line; ellipsis never splits a Unicode grapheme. */
export function fitGraphLabel(value,maxWidth,measureText) {
  const normalized=cleanText(value),original=normalized.text;
  if (!original || !finite(maxWidth) || maxWidth <= 0 || typeof measureText !== 'function') return {text:'',width:0,fullText:false};
  const width=measured(original,measureText);
  if (width <= maxWidth && !normalized.truncated) return {text:original,width,fullText:true};
  const ellipsis='…',ellipsisWidth=measured(ellipsis,measureText);
  if (!finite(ellipsisWidth) || ellipsisWidth > maxWidth) return {text:'',width:0,fullText:false};
  const parts=graphemes(original);
  // A short extension is useful when several uploaded files share the same title.
  // Reserve it only when enough room remains for a recognizable filename prefix.
  const extension=original.match(/\.[A-Za-z][A-Za-z0-9]{0,7}$/u)?.[0] || '';
  const suffix=extension && measured(`…${extension}`,measureText) <= maxWidth*.6 ? extension : '';
  const tail=`${ellipsis}${suffix}`;
  const prefixParts=suffix ? graphemes(original.slice(0,-suffix.length)) : parts;
  let lo=0,hi=prefixParts.length;
  while (lo < hi) {
    const mid=Math.ceil((lo+hi)/2);
    if (measured(`${prefixParts.slice(0,mid).join('')}${tail}`,measureText) <= maxWidth) lo=mid;
    else hi=mid-1;
  }
  const text=`${prefixParts.slice(0,lo).join('')}${tail}`;
  const fittedWidth=measured(text,measureText);
  return finite(fittedWidth) && fittedWidth <= maxWidth ? {text,width:fittedWidth,fullText:false} : {text:'',width:0,fullText:false};
}

function occupancyGrid(width,height) {
  const cells=new Map(),large=[],cell=GRAPH_LABEL_LIMITS.cell;
  function cellsFor(bounds) {
    const x0=Math.floor(Math.max(0,bounds.x)/cell),y0=Math.floor(Math.max(0,bounds.y)/cell);
    const x1=Math.floor(Math.min(width,bounds.x+bounds.width)/cell),y1=Math.floor(Math.min(height,bounds.y+bounds.height)/cell);
    return {x0,y0,x1,y1,count:Math.max(0,x1-x0+1)*Math.max(0,y1-y0+1)};
  }
  return {
    add(obstacle,bounds) {
      const range=cellsFor(bounds);
      // Extreme input radii must not allocate an unbounded number of grid cells.
      if (range.count > 256) {large.push(obstacle);return;}
      for (let x=range.x0;x<=range.x1;x++) for (let y=range.y0;y<=range.y1;y++) {
        const id=key(x,y),items=cells.get(id);
        if (items) items.push(obstacle); else cells.set(id,[obstacle]);
      }
    },
    query(rect) {
      const range=cellsFor(rect),found=new Set(large);
      for (let x=range.x0;x<=range.x1;x++) for (let y=range.y0;y<=range.y1;y++) {
        for (const item of cells.get(key(x,y)) || []) found.add(item);
      }
      return found;
    }
  };
}

function overlaps(rect,obstacle) {
  if (obstacle.kind === 'disk') {
    const x=Math.max(rect.x,Math.min(rect.x+rect.width,obstacle.x));
    const y=Math.max(rect.y,Math.min(rect.y+rect.height,obstacle.y));
    return (x-obstacle.x)**2+(y-obstacle.y)**2 < obstacle.radius**2;
  }
  return rect.x < obstacle.x+obstacle.width && rect.x+rect.width > obstacle.x && rect.y < obstacle.y+obstacle.height && rect.y+rect.height > obstacle.y;
}

function anchorPosition(node,width,height,anchor,gap) {
  const r=node.radius+gap;
  if (anchor === 'e') return {x:node.x+r,y:node.y-height/2};
  if (anchor === 'w') return {x:node.x-r-width,y:node.y-height/2};
  if (anchor === 'n') return {x:node.x-width/2,y:node.y-r-height};
  if (anchor === 's') return {x:node.x-width/2,y:node.y+r};
  return {x:anchor.endsWith('e') ? node.x+r : node.x-r-width,y:anchor.startsWith('n') ? node.y-r-height : node.y+r};
}

/**
 * nodes use projected screen pixels (radius included). `priority` orders neighbors
 * after selected and route nodes; `degree` breaks remaining ties before stable ID.
 * Returned x/y are the outer pill top-left; textX/textY use a middle baseline.
 * Previous output rows, keyed by ID, preserve anchors without random label jitter.
 */
export function layoutGraphLabels({nodes=[],width,height,measureText,previous=new Map(),maxLabels=32,fontSize=12,selectedId='',routeIds=[]}={}) {
  if (!finite(width) || !finite(height) || width <= 0 || height <= 0 || width > GRAPH_LABEL_LIMITS.viewportMax || height > GRAPH_LABEL_LIMITS.viewportMax || !finite(fontSize) || fontSize <= 0 || typeof measureText !== 'function') return [];
  const inset=GRAPH_LABEL_LIMITS.inset,clearance=GRAPH_LABEL_LIMITS.clearance;
  const paddingX=7,paddingY=4,labelHeight=Math.ceil(fontSize*1.3)+paddingY*2;
  if (width <= inset*2+paddingX*2 || height < inset*2+labelHeight) return [];
  const limit=finite(maxLabels) ? Math.min(GRAPH_LABEL_LIMITS.labels,Math.max(0,Math.floor(maxLabels))) : 0;
  if (!limit) return [];
  const seen=new Set(),visible=[];
  for (const value of (Array.isArray(nodes) ? nodes : []).slice(0,GRAPH_LABEL_LIMITS.nodes)) {
    if (!value || typeof value.id !== 'string' || !value.id || seen.has(value.id) || !finite(value.x) || !finite(value.y)) continue;
    const radius=value.radius === undefined ? 6 : value.radius;
    if (!finite(radius) || radius <= 0) continue;
    seen.add(value.id);
    if (value.x+radius < 0 || value.x-radius > width || value.y+radius < 0 || value.y-radius > height) continue;
    visible.push({id:value.id,label:typeof value.label === 'string' ? value.label : '',x:value.x,y:value.y,radius,priority:finite(value.priority)?value.priority:0,degree:finite(value.degree)?value.degree:0});
  }
  const route=new Set(Array.isArray(routeIds) ? routeIds : routeIds instanceof Set ? routeIds : []);
  const tier=node=>node.id === selectedId ? 2 : route.has(node.id) ? 1 : 0;
  visible.sort((a,b)=>tier(b)-tier(a)||b.priority-a.priority||b.degree-a.degree||compareIds(a.id,b.id));
  const grid=occupancyGrid(width,height);
  for (const node of visible) {
    const radius=node.radius+clearance;
    grid.add({kind:'disk',x:node.x,y:node.y,radius},{x:node.x-radius,y:node.y-radius,width:radius*2,height:radius*2});
  }
  const result=[];
  for (const node of visible.slice(0,GRAPH_LABEL_LIMITS.candidates)) {
    if (!node.label) continue;
    const maxTextWidth=Math.min(node.id === selectedId ? 260 : width < 600 ? 120 : 160,width-inset*2-paddingX*2);
    const fit=fitGraphLabel(node.label,maxTextWidth,measureText);
    if (!fit.text) continue;
    const labelWidth=Math.ceil(fit.width)+paddingX*2;
    const old=previous instanceof Map ? previous.get(node.id) : null;
    const order=old && ANCHORS.includes(old.anchor) ? [old.anchor,...ANCHORS.filter(anchor=>anchor !== old.anchor)] : ANCHORS;
    // A focused label can move out of a dense cluster instead of disappearing.
    // Its nodeX/nodeY and gap let the renderer add a restrained leader if needed.
    const gaps=node.id === selectedId ? [7,19,35,59,91,139] : route.has(node.id) ? [7,19] : [7];
    const placements=[];
    if (old && ANCHORS.includes(old.anchor) && finite(old.gap) && gaps.includes(old.gap)) placements.push({anchor:old.anchor,gap:old.gap});
    for (const gap of gaps) for (const anchor of order) placements.push({anchor,gap});
    const tested=new Set();
    for (const {anchor,gap} of placements) {
      const location=anchorPosition(node,labelWidth,labelHeight,anchor,gap);
      // Inward clamping is accepted only if it also clears the node disks below.
      const rect={x:Math.max(inset,Math.min(width-inset-labelWidth,location.x)),y:Math.max(inset,Math.min(height-inset-labelHeight,location.y)),width:labelWidth,height:labelHeight};
      const signature=`${rect.x},${rect.y}`;
      if (tested.has(signature)) continue;
      tested.add(signature);
      if ([...grid.query(rect)].some(obstacle=>overlaps(rect,obstacle))) continue;
      result.push({id:node.id,text:fit.text,fullText:fit.fullText,lines:[fit.text],...rect,anchor,gap,paddingX,paddingY,textX:rect.x+paddingX,textY:rect.y+rect.height/2,nodeX:node.x,nodeY:node.y});
      grid.add({kind:'rect',x:rect.x-clearance,y:rect.y-clearance,width:rect.width+clearance*2,height:rect.height+clearance*2},
        {x:rect.x-clearance,y:rect.y-clearance,width:rect.width+clearance*2,height:rect.height+clearance*2});
      break;
    }
    if (result.length >= limit) break;
  }
  return result;
}
