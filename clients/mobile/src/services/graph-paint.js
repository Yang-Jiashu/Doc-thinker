// Screen-space presentation only: this renderer never changes graph topology,
// source records, physics, or hit-testing. All dimensions below are CSS pixels.
import {layoutGraphLabels} from './graph-label-layout.js';
import {buildEdgeRoutes} from './graph-edge-geometry.js';

export const GRAPH_FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif';
export const GRAPH_COLORS = Object.freeze(['#35665a','#5c789d','#a17b48','#907294','#6d8451','#9a6860']);
const clamp=(value,low,high)=>Math.max(low,Math.min(high,value));

export function createGraphPainter(){
  let previous=new Map();
  const measurements=new Map();
  return {
    reset(){previous=new Map();measurements.clear();},
    paint(ctx,{scene,view,width,height,kinds=[],selectedId='',highlightIds=[]}){
      const fontSize=width<500 ? 12 : 12.5, font=`600 ${fontSize}px ${GRAPH_FONT}`;
      ctx.font=font;ctx.fontKerning='normal';
      const measure=text=>{
        const key=font+'\n'+text;
        if(measurements.has(key))return measurements.get(key);
        const value=ctx.measureText(text).width;
        if(measurements.size>=1024)measurements.clear();
        measurements.set(key,value);return value;
      };
      const route=new Set(highlightIds),neighbors=scene.adjacency.get(selectedId)||new Set();
      const emphasized=id=>id===selectedId||route.has(id)||neighbors.has(id);
      const points=new Map(),projected=[];
      for(const node of scene.nodes){
        const point={id:node.id,label:node.label,type:node.type,degree:node.degree,
          x:node.x*view.k+view.x,y:node.y*view.k+view.y,
          radius:clamp(4.5+Math.sqrt(node.degree)*.9,4.5,8.5)*clamp(Math.sqrt(view.k),.85,1.4),
          priority:neighbors.has(node.id)?100:0};
        points.set(node.id,point);projected.push(point);
      }
      const labels=layoutGraphLabels({nodes:projected,width,height,measureText:measure,previous,
        fontSize,maxLabels:width<500?16:32,selectedId,routeIds:highlightIds});
      previous=new Map(labels.map(row=>[row.id,row]));
      const routes=buildEdgeRoutes(scene.edges,points);
      const focus=edge=>edge.source.id===selectedId||edge.target.id===selectedId||(route.has(edge.source.id)&&route.has(edge.target.id));
      const drawEdge=path=>{
        const focused=focus(path.edge),inferred=path.edge.is_discovered;
        ctx.globalAlpha=focused ? .95 : selectedId ? .34 : .72;
        ctx.strokeStyle=focused?'#557b69':inferred?'#b29a75':'#b3c0b2';
        ctx.lineWidth=focused?1.55:.85;ctx.setLineDash(inferred?[3,5]:[]);
        ctx.beginPath();ctx.moveTo(path.start.x,path.start.y);
        if(path.control)ctx.quadraticCurveTo(path.control.x,path.control.y,path.end.x,path.end.y);
        else ctx.lineTo(path.end.x,path.end.y);
        ctx.stroke();
        if(focused&&path.arrow?.length===3){
          ctx.setLineDash([]);ctx.fillStyle=ctx.strokeStyle;ctx.beginPath();
          path.arrow.forEach((point,index)=>index?ctx.lineTo(point.x,point.y):ctx.moveTo(point.x,point.y));
          ctx.closePath();ctx.fill();
        }
      };
      // Reserve quiet ink-free space behind captions, without putting every
      // label in a box. The even-odd holes cannot overlap (layout contract).
      ctx.save();ctx.beginPath();ctx.rect(0,0,width,height);
      for(const label of labels)ctx.rect(label.x,label.y,label.width,label.height);
      ctx.clip('evenodd');ctx.lineCap='round';ctx.lineJoin='round';
      for(const path of routes)if(!focus(path.edge))drawEdge(path);
      for(const path of routes)if(focus(path.edge))drawEdge(path);
      ctx.restore();ctx.setLineDash([]);
      const ranks=new Map(kinds.map((kind,index)=>[kind,index]));
      for(const node of projected){
        if(node.x<-20||node.x>width+20||node.y<-20||node.y>height+20)continue;
        const selected=node.id===selectedId,focused=emphasized(node.id);
        ctx.globalAlpha=selectedId&&!focused ? .6 : 1;
        const color=GRAPH_COLORS[(ranks.get(node.type||'知识点')||0)%GRAPH_COLORS.length];
        if(selected){ctx.fillStyle='#e5efda';ctx.beginPath();ctx.arc(node.x,node.y,node.radius+5,0,Math.PI*2);ctx.fill();}
        ctx.fillStyle=selected?'#294f43':color;ctx.beginPath();ctx.arc(node.x,node.y,node.radius,0,Math.PI*2);ctx.fill();
        ctx.strokeStyle='#fffefa';ctx.lineWidth=1.5;ctx.stroke();
      }
      ctx.globalAlpha=1;ctx.textAlign='left';ctx.textBaseline='middle';
      for(const label of labels){
        const selected=label.id===selectedId;
        if(label.gap>19){
          const node=points.get(label.id),x=clamp(node.x,label.x,label.x+label.width),y=clamp(node.y,label.y,label.y+label.height);
          const dx=x-node.x,dy=y-node.y,length=Math.hypot(dx,dy);
          if(length>node.radius+2){ctx.strokeStyle='#aebdaa';ctx.lineWidth=.7;ctx.setLineDash([2,3]);ctx.beginPath();ctx.moveTo(node.x+dx/length*(node.radius+2),node.y+dy/length*(node.radius+2));ctx.lineTo(x,y);ctx.stroke();ctx.setLineDash([]);}
        }
        ctx.font=`${selected?'600':'400'} ${fontSize}px ${GRAPH_FONT}`;
        ctx.fillStyle=selected?'#23483a':selectedId&&!emphasized(label.id)?'#788879':'#4e6357';
        ctx.fillText(label.text,label.textX,label.textY);
      }
      return {labels,nodes:projected.length,edges:routes.length};
    },
  };
}
