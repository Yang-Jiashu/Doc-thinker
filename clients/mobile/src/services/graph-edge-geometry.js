// Display-only geometry in CSS pixels. Routes retain their original knowledge edge.
// Topology is immutable during a scene's life; coordinates are deliberately not cached.
const LIMITS = Object.freeze({edges:8000,points:2000,lane:18,maxLane:36,padding:2});
const plans = new WeakMap();
const finite = value => typeof value === 'number' && Number.isFinite(value);
const endpoint = value => typeof value === 'string' ? value : value && typeof value.id === 'string' ? value.id : '';
const compare = (a,b) => a < b ? -1 : a > b ? 1 : 0;
const text = value => typeof value === 'string' ? value : '';

function prepare(edges) {
  if (plans.has(edges)) return plans.get(edges);
  const pairs=new Map();
  for (const edge of edges.slice(0,LIMITS.edges)) {
    if (!edge || typeof edge !== 'object') continue;
    const source=endpoint(edge.source),target=endpoint(edge.target);
    if (!source || !target || source===target) continue;
    const low=source<target?source:target,high=source<target?target:source;
    const pair=JSON.stringify([low,high]);
    if (!pairs.has(pair)) pairs.set(pair,[]);
    pairs.get(pair).push({edge,source,target,low,high});
  }
  const plan=[];
  for (const key of [...pairs.keys()].sort(compare)) {
    const group=pairs.get(key);
    group.sort((a,b)=>compare(a.source,b.source)||compare(a.target,b.target)||
      compare(text(a.edge.description),text(b.edge.description))||
      compare(text(a.edge.relation),text(b.edge.relation))||
      compare(text(a.edge.type),text(b.edge.type))||
      compare(text(a.edge.id),text(b.edge.id))||
      Number(a.edge.directed===true)-Number(b.edge.directed===true));
    const half=(group.length-1)/2;
    const step=half>0?Math.min(LIMITS.lane,LIMITS.maxLane/half):0;
    group.forEach((item,index)=>plan.push({...item,lane:(index-half)*step}));
  }
  plans.set(edges,plan);
  return plan;
}

function validPoint(point) {
  return point && finite(point.x) && finite(point.y) && finite(point.radius) && point.radius>=0;
}

// The symmetric centre-to-centre quadratic has local coordinates
// (distance*t, 2*offset*t*(1-t)). Solve its exact intersection with a node circle.
// The upper bound radius/distance makes precision independent of viewport zoom.
function clipFraction(radius,distance,offset) {
  const ratio=radius/distance,h=offset/distance;
  if (!h || !ratio) return ratio;
  // |h| <= .35 makes this radial polynomial strictly increasing. Its linear
  // tangent is a close lower-bound seed; fixed Newton steps avoid 44 bisections
  // per curved edge on every animation frame, without pixel-dependent tolerances.
  const h4=4*h*h,square=ratio*ratio;
  let t=ratio/Math.sqrt(1+h4);
  for(let i=0;i<5;i++) {
    const u=1-t,f=t*t*(1+h4*u*u)-square;
    const derivative=2*t*(1+h4*u*(1-2*t));
    if (!derivative) break;
    t=Math.max(0,Math.min(ratio,t-f/derivative));
  }
  return t;
}

const quadratic = (a,c,b,t) => {
  const u=1-t;
  return {x:u*u*a.x+2*u*t*c.x+t*t*b.x,y:u*u*a.y+2*u*t*c.y+t*t*b.y};
};

/**
 * Make bounded, deterministic screen-space paths. An ordinary edge stays straight;
 * only real parallel/reciprocal links fan into distinct, gently curved lanes.
 * points: Map<nodeId, {x,y,radius}>; radius and all returned values are CSS pixels.
 * Treat edges as immutable (replace its array when topology/edge metadata changes).
 */
export function buildEdgeRoutes(edges,points) {
  if (!Array.isArray(edges) || !(points instanceof Map)) return [];
  const allowed=new Map();
  let count=0;
  for(const [id,point] of points) {
    if (count++>=LIMITS.points) break;
    if (typeof id==='string' && validPoint(point)) allowed.set(id,point);
  }
  const routes=[];
  for(const item of prepare(edges)) {
    const a=allowed.get(item.source),b=allowed.get(item.target);
    if (!a || !b) continue;
    const dx=b.x-a.x,dy=b.y-a.y,distance=Math.hypot(dx,dy);
    const ra=a.radius+LIMITS.padding,rb=b.radius+LIMITS.padding;
    if (!finite(distance) || distance<=ra+rb+.5) continue;
    const ux=dx/distance,uy=dy/distance;
    let start,end,control=null;
    if (!item.lane) {
      start={x:a.x+ux*ra,y:a.y+uy*ra};
      end={x:b.x-ux*rb,y:b.y-uy*rb};
    } else {
      // Use the same normal for both directions, so reciprocal links cannot overlap.
      const canonical=item.source===item.low?1:-1;
      const offset=Math.sign(item.lane)*Math.min(Math.abs(item.lane)*2,distance*.35);
      const c={x:(a.x+b.x)/2-uy*canonical*offset,y:(a.y+b.y)/2+ux*canonical*offset};
      if (!finite(c.x) || !finite(c.y)) continue;
      const t0=clipFraction(ra,distance,offset),t1=1-clipFraction(rb,distance,offset);
      if (t1<=t0) continue;
      start=quadratic(a,c,b,t0);end=quadratic(a,c,b,t1);
      const span=t1-t0;
      control={x:start.x+span*((1-t0)*(c.x-a.x)+t0*(b.x-c.x)),
        y:start.y+span*((1-t0)*(c.y-a.y)+t0*(b.y-c.y))};
    }
    const mid=control?quadratic(start,control,end,.5):{x:(start.x+end.x)/2,y:(start.y+end.y)/2};
    const tangent={x:end.x-(control||start).x,y:end.y-(control||start).y};
    const length=Math.hypot(tangent.x,tangent.y),visible=Math.hypot(end.x-start.x,end.y-start.y);
    if (![start,end,mid,control].filter(Boolean).every(p=>finite(p.x)&&finite(p.y)) || !finite(length) || !finite(visible)) continue;
    let arrow=null;
    if (item.edge.directed===true && visible>=12 && length>0) {
      const tx=tangent.x/length,ty=tangent.y/length;
      arrow=[{...end},{x:end.x-tx*7-ty*2,y:end.y-ty*7+tx*2},
        {x:end.x-tx*7+ty*2,y:end.y-ty*7-tx*2}];
    }
    routes.push({edge:item.edge,start,end,control,arrow,mid});
  }
  return routes;
}
