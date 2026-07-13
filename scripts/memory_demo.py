#!/usr/bin/env python3
"""DocThinker Agentic Memory Demo."""
from __future__ import annotations
import asyncio,hashlib,json,math,os,re,shutil,sys,tempfile,time
from pathlib import Path
from typing import Any,Dict,List,Optional,Sequence,Tuple
_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(_ROOT))
from claw import ClawMemoryManager,MemoryConfig
from neuro_memory import MemoryEngine,spreading_activation
from neuro_memory.consolidation import consolidate
from docthinker.memory_core import AgentMemoryBackends,AgentMemoryCore,InMemoryLongHorizonBackend,MemoryPolicy
from docthinker.memory_core.adapters import ClawConversationBackend,NeuroEpisodicBackend
B="\033[1m";D="\033[2m";R="\033[31m";G="\033[32m";Y="\033[33m";BL="\033[34m";M="\033[35m";C="\033[36m";X="\033[0m"
def header(t):
    bar="="*72
    print(f"\n{B}{C}{bar}{X}")
    print(f"{B}{C}  {t}{X}")
    print(f"{B}{C}{bar}{X}")
def step(n,t):
    bar="-"*72
    print(f"\n{B}{Y}{bar}{X}")
    print(f"{B}{Y}  Step {n}: {t}{X}")
    print(f"{B}{Y}{bar}{X}")
def lbl(t,c=D):
    print(f"{c}{B}  > {t}{X}")
def kv(k,v,i=6):
    print(f"{D}{chr(32)*i}{k}:{X} {v}")
def blk(title,lines,c=""):
    p=c or ""
    d="-"*max(1,60-len(title))
    print(f"{p}  +-- {title} {d}{X}")
    for l in lines:
        print(f"{p}  | {l}{X}")
    print(f"{p}  +{chr(45)*71}{X}")
class MockEmb:
    @staticmethod
    def _tok(t):
        t=t.lower()
        toks=re.findall(r"[\u4e00-\u9fff]{1,4}|[a-z]{2,}",t)
        for m in re.finditer(r"[\u4e00-\u9fff]{2,}",t):
            s=m.group(0)
            for i in range(len(s)-1):toks.append(s[i:i+2])
        return toks
    @staticmethod
    def _h(t,d):return int(hashlib.md5(t.encode()).hexdigest(),16)%d
    def __call__(self,texts):
        d=128;res=[]
        for t in texts:
            v=[0.0]*d
            for tok in self._tok(t):
                v[self._h(tok,d)]+=1.0;v[self._h(tok+"#2",d)]+=0.5
            n=math.sqrt(sum(x*x for x in v)) or 1e-8
            res.append([x/n for x in v])
        return res

class MockLLM:
    def __init__(self):self.cc=0
    async def __call__(self,prompt,**kw):
        self.cc+=1;pl=prompt.lower()
        if "memory.md" in pl and ("update" in pl or "create" in pl or "memory" in pl):return self._md(prompt)
        if "compare" in pl or "analogy" in pl or "relation" in pl:return "relation: same_theme\nreason: both episodes discuss memory system"
        return f"[mock-llm] processed ({len(prompt)} chars)"
    def _md(self,prompt):
        ls=["# Core Memory",""];found=[]
        pm={"spark":"Spark Research Assistant project","seven-day":"Seven-day task continuation metric","plan b":"Plan B long-term memory: 12 dev-days","plan a":"Plan A knowledge star map: 20 dev-days","lin xia":"PM Lin Xia supports Plan B","zhou heng":"Designer Zhou Heng supports Plan A","chen mo":"Eng lead Chen Mo proposed hybrid plan","constraint":"Memory must be visible/toggleable/editable/deletable"}
        pl=prompt.lower()
        for k,v in pm.items():
            if k in pl:found.append(v)
        ls.append("## Key Decisions")
        for f in found:ls.append(f"- {f}")
        if not found:ls.append("- (waiting for more conversation)")
        ls+=["","## User Profile","- Research assistant scenario, focused on long-term task continuity","","## Active Topics","- Product plan selection and seven-day task continuation rate",""]
        return "\n".join(ls)
class DemoClaw:
    def __init__(self,td,lf,ef):
        self._m=ClawMemoryManager(talk_dir=td,llm_func=lf,embedding_func=ef,config=MemoryConfig(working_memory_turns=4,core_memory_update_interval=3,archive_chunk_size=400,archive_top_k=3,archive_min_score=0.15))
        self.td=td
    async def build_memory_context(self,q,*,enable_archive=True):return await self._m.build_memory_context(q,enable_archive=enable_archive)
    async def post_query_update(self,q,a,sid,ts):
        tf=Path(self.td)/"talk.json";msgs=[]
        if tf.exists():
            try:msgs=json.loads(tf.read_text("utf-8")).get("messages",[])
            except:msgs=[]
        msgs.append({"role":"user","content":q,"timestamp":ts})
        msgs.append({"role":"assistant","content":a,"timestamp":ts})
        tf.parent.mkdir(parents=True,exist_ok=True)
        tf.write_text(json.dumps({"messages":msgs},ensure_ascii=False,indent=2),"utf-8")
        await self._m.post_query_update(q,a,sid,ts)
    def get_stats(self):return self._m.get_stats()

class DemoNeuro:
    def __init__(self,ef,lf,wd):self.eng=MemoryEngine(embedding_func=ef,llm_func=lf,working_dir=wd)
    async def add_observation(self,**kw):return await self.eng.add_observation(**kw)
    async def retrieve_analogies(self,q,*,top_k=5,then_spread=True,spread_top_k=3):return await self.eng.retrieve_analogies(q,top_k=top_k,then_spread=then_spread,spread_top_k=spread_top_k)
    @property
    def graph(self):return self.eng.graph
    @property
    def episode_store(self):return self.eng.episode_store
    async def run_consolidation(self):
        eps=self.episode_store.all_episodes()
        async def csf(a,b):
            ea,eb=eps.get(a),eps.get(b)
            if not ea or not eb or not ea.content_embedding or not eb.content_embedding:return 0.0
            va,vb=ea.content_embedding,eb.content_embedding
            dot=sum(x*y for x,y in zip(va,vb))
            na=math.sqrt(sum(x*x for x in va)) or 1e-8
            nb=math.sqrt(sum(x*x for x in vb)) or 1e-8
            return dot/(na*nb)
        return await consolidate(self.graph,eps,recent_n=50,high_salience_n=20,content_sim_threshold=0.15,structure_sim_threshold=0.1,llm_func=self.eng.llm_func,content_sim_fn=csf)
async def run_demo():
    tmp=Path(tempfile.mkdtemp(prefix="dt_mem_demo_"))
    print(f"{D}  [workspace] {tmp}{X}")
    try:
        header("DocThinker Agentic Memory Demo")
        print("\n  This demo simulates a multi-turn research conversation")
        print("  and shows how each memory layer works together.\n")
        print("  Memory layers demonstrated:")
        print(f"    {G}Claw{X}      - Hot / Warm / Cold conversation memory")
        print(f"    {M}Neuro{X}     - Episodic + spreading activation + analogy")
        print(f"    {BL}Long-Horizon{X} - Durable insights with write/skip decisions")
        print(f"    {Y}Core{X}      - AgentMemoryCore unified recall -> consolidate")
        ef=MockEmb();lf=MockLLM()
        td=str(tmp/"talk");nd=str(tmp/"neuro")
        os.makedirs(td,exist_ok=True)
        claw=DemoClaw(td,lf,ef);neuro=DemoNeuro(ef,lf,nd);lh=InMemoryLongHorizonBackend()
        mem=AgentMemoryCore(backends=AgentMemoryBackends(conversation=ClawConversationBackend(lambda sid:claw),episodic=NeuroEpisodicBackend(lambda sid:neuro),long_horizon=lh),policy=MemoryPolicy(episodic_top_k=3,long_horizon_top_k=3,long_horizon_min_confidence=0.30,enabled_layers=("conversation","episodic","long_horizon")))
        turns=[
            {"q":"We are building Spark Research Assistant. Core metric is seven-day task continuation rate. Should we prioritize knowledge star map or long-term memory?","a":"Based on your target metric, I recommend Plan B long-term memory. User interviews show the core pain point is re-explaining background each time. Plan B needs 12 dev-days. Plan A has visual impact but only 2/6 testers found evidence via the map.","c":["Spark","seven-day","Plan B","long-term memory","Plan A","star map"]},
            {"q":"What do Lin Xia, Zhou Heng, and Chen Mo each support? What is Chen Mo compromise?","a":"PM Lin Xia supports Plan B. Designer Zhou Heng supports Plan A for visual impact. Eng lead Chen Mo proposes: 12 days Plan B, 6 days simplified map, 4 days testing. Star map relations must open source evidence.","c":["Lin Xia","Zhou Heng","Chen Mo","Plan B","Plan A","compromise"]},
            {"q":"Please remember: long-term memory must be visible, toggleable, editable, and deletable. This is a design constraint.","a":"Recorded. The system long-term memory must be visible, toggleable, editable, and deletable. MEMORY.md should be exportable and auditable. Long-term records support list, update, and delete.","c":["long-term memory","visible","toggleable","editable","deletable","constraint"]},
            {"q":"Based on memory constraints we discussed, how should seven-day continuation rate be implemented?","a":"Implementation needs: 1) Auto-recall previous context on new conversation; 2) Save why directions were excluded; 3) Memory must be auditable and deletable for trust. Plan B compromise makes visibility key. Show which historical info was used each turn.","c":["seven-day","long-term memory","Plan B","excluded direction","auditable"]},
        ]
        for i,t in enumerate(turns,1):
            step(i,f"Conversation Turn {i}")
            print()
            qs=t["q"][:120]+("..." if len(t["q"])>120 else "")
            lbl(f"User: {qs}",G);print()
            lbl("Recalling memory context...",BL)
            bundle=await mem.recall(session_id="demo-session",query=t["q"],enable_thinking=True)
            tr=bundle.trace
            rl=[f"memory_mode: {tr.memory_mode}",f"memory_hits: {tr.memory_hits}",f"episodic_hits: {tr.episodic_hits}",f"long_horizon_hits: {tr.long_horizon_hits}",f"context_injected: {tr.memory_context_injected}"]
            if tr.recall_plan:
                rl.append("recall_plan.question_type: "+str(tr.recall_plan.get("question_type","?")))
                rl.append("recall_plan.layers: "+str(tr.recall_plan.get("layers",[])))
            if tr.memory_reasoning:
                rl.append("memory_reasoning.conclusions: "+str(tr.memory_reasoning.get("conclusions",[])))
            blk("Memory Trace (recall)",rl,BL)
            if bundle.episodic_matches:
                el=[f"[{m.get(chr(115)+chr(99)+chr(111)+chr(114)+chr(101),0):.2f}] {m.get(chr(115)+chr(117)+chr(109)+chr(109)+chr(97)+chr(114)+chr(121),chr(32))[:100]}" for m in bundle.episodic_matches[:3]]
                blk("Episodic Matches",el,M)
            if bundle.long_horizon_matches:
                ll=[f"[{m.get(chr(115)+chr(99)+chr(111)+chr(112)+chr(101),chr(63))}/{m.get(chr(107)+chr(105)+chr(110)+chr(100),chr(63))}/{m.get(chr(99)+chr(111)+chr(110)+chr(102)+chr(105)+chr(100)+chr(101)+chr(110)+chr(99)+chr(101),0):.2f}] {m.get(chr(115)+chr(117)+chr(109)+chr(109)+chr(97)+chr(114)+chr(121),chr(32))[:100]}" for m in bundle.long_horizon_matches[:3]]
                blk("Long-Horizon Matches",ll,Y)
            if bundle.retrieval_instruction:
                ip=bundle.retrieval_instruction[:400]+("..." if len(bundle.retrieval_instruction)>400 else "")
                blk("Retrieval Instruction (merged)",[ip],D)
            print()
            a_s=t["a"][:120]+("..." if len(t["a"])>120 else "")
            lbl(f"Assistant: {a_s}",C);print()
            lbl("Consolidating memory...",Y)
            res=await mem.after_response(session_id="demo-session",question=t["q"],answer=t["a"],matched_expanded=bundle.expanded_matches)
            cl=[f"claw_updated: {res.get(chr(99)+chr(108)+chr(97)+chr(119)+chr(95)+chr(117)+chr(112)+chr(100)+chr(97)+chr(116)+chr(101)+chr(100))}",f"episode_added: {res.get(chr(101)+chr(112)+chr(105)+chr(115)+chr(111)+chr(100)+chr(101)+chr(95)+chr(97)+chr(100)+chr(100)+chr(101)+chr(100))}",f"long_horizon_insight_added: {res.get(chr(108)+chr(111)+chr(110)+chr(103)+chr(95)+chr(104)+chr(111)+chr(114)+chr(105)+chr(122)+chr(111)+chr(110)+chr(95)+chr(105)+chr(110)+chr(115)+chr(105)+chr(103)+chr(104)+chr(116)+chr(95)+chr(97)+chr(100)+chr(100)+chr(101)+chr(100))}"]
            wd=res.get("long_horizon_write_decision",{})
            if wd:
                cl.append("write_decision.action: "+str(wd.get("action")))
                cl.append("write_decision.reason: "+str(wd.get("reason")))
            if res.get("long_horizon_insight"):
                ins=res["long_horizon_insight"]
                cl.append("insight.kind: "+str(ins.get("kind")))
                cl.append("insight.scope: "+str(ins.get("scope")))
                cl.append("insight.confidence: "+str(ins.get("confidence")))
                cl.append("insight.summary: "+str(ins.get("summary",""))[:100])
            blk("Consolidation Result",cl,Y)
        step(5,"Memory State After 4 Turns")
        print()
        lbl("Claw Three-Tier Memory:",G)
        cs=claw.get_stats()
        kv("working_memory_turns",str(cs.get("working_turns",0)))
        kv("core_memory_exists",str(cs.get("core_memory_exists",False)))
        kv("core_memory_bytes",str(cs.get("core_memory_bytes",0)))
        kv("archive_chunks",str(cs.get("archive_chunks",0)))
        mmd=claw._m.core.read()
        if mmd:
            print()
            blk("MEMORY.md (Core Memory)",mmd.strip().split("\n"),G)
        print()
        lbl("Neuro Memory (Episodic):",M)
        eps=neuro.episode_store.all_episodes()
        kv("total_episodes",str(len(eps)))
        kv("graph_nodes",str(len(neuro.graph.get_all_nodes())))
        kv("graph_edges",str(len(neuro.graph.get_all_edges())))
        for eid,ep in list(eps.items())[:4]:
            print(f"        {D}-{X} {ep.episode_id[:20]} | concepts: {ep.concepts[:4]} | retr: {ep.retrieval_count}")
        print()
        lbl("Long-Horizon Memory:",Y)
        ls=lh.stats(session_id="demo-session")
        kv("total_insights",str(ls.get("count",0)))
        kv("by_kind",str(ls.get("by_kind",{})))
        kv("last_write_decision",str(ls.get("last_write_decision",{})))
        print()
        ins_list=lh.list_insights(session_id="demo-session",limit=10)
        if ins_list:
            for it in ins_list:
                print(f"        {D}-{X} [{it.get(chr(115)+chr(99)+chr(111)+chr(112)+chr(101))}/{it.get(chr(107)+chr(105)+chr(110)+chr(100))}/{it.get(chr(99)+chr(111)+chr(110)+chr(102)+chr(105)+chr(100)+chr(101)+chr(110)+chr(99)+chr(101),0):.2f}] {it.get(chr(115)+chr(117)+chr(109)+chr(109)+chr(97)+chr(114)+chr(121),chr(32))[:80]}")
        step(6,"Spreading Activation (Neuro Memory)")
        print("\n  Simulating: user mentions memory,")
        print("  spreading activation propagates through the graph.\n")
        seeds=[]
        for nid,node in neuro.graph._nodes.items():
            if "memory" in json.dumps(node,ensure_ascii=False).lower():
                seeds.append(nid)
        if not seeds and neuro.graph._nodes:
            seeds=list(neuro.graph._nodes.keys())[:3]
        if seeds:
            lbl(f"Seed nodes: {seeds[:5]}",M)
            act=spreading_activation(neuro.graph,seeds,max_hops=3,initial_activation=1.0)
            if act:
                al=[f"[{s:.4f}] {nid[:35]} ({(neuro.graph.get_node(nid) or {}).get(chr(116)+chr(121)+chr(112)+chr(101),chr(63))})" for nid,s in act[:10]]
                blk("Activated Nodes (top 10)",al,M)
            else:
                print(f"  {D}(no nodes activated above threshold){X}")
        else:
            print(f"  {D}(no seed nodes found){X}")
        step(7,"Memory Consolidation (Cross-Episode Inference)")
        print("\n  Running consolidation: replay episodes, infer relations, strengthen edges.\n")
        cres=await neuro.run_consolidation()
        lbl(f"Result: {cres}",M);print()
        kv("edges_added",str(cres.get("edges_added",0)))
        kv("pairs_processed",str(cres.get("pairs_processed",0)))
        kv("edges_strengthened",str(cres.get("edges_strengthened",0)))
        aedges=neuro.graph.get_all_edges()
        cedges=[e for e in aedges if e.edge_type.value in ("analogous_to","same_theme","episode_similarity")]
        if cedges:
            print()
            lbl("Cross-Episode Relations Inferred:",M)
            for e in cedges[:8]:
                print(f"        {D}-{X} {e.source_id[:25]} {B}--[{e.edge_type.value}]->{X} {e.target_id[:25]} (w={e.weight:.2f})")
        step(8,"Memory Editing & Audit Export")
        print()
        lbl("Natural-language edit: delete memory about Plan A",R)
        ep=lh.plan_edit(session_id="demo-session",instruction="delete memory about Plan A")
        el=["inferred_action: "+str(ep.get("action")),"candidates_found: "+str(len(ep.get("candidates",[])))]
        for c in ep.get("candidates",[])[:3]:
            el.append("  ["+str(c.get("match",{}).get("score",0))+"] "+str(c.get("summary",""))[:80])
        blk("Edit Plan",el,R)
        print()
        lbl("Exporting long-horizon memory as MEMORY.md...",Y)
        exp=lh.export_markdown(session_id="demo-session")
        elns=exp.strip().split("\n")
        blk("Exported MEMORY.md (Long-Horizon)",elns[:30],Y)
        if len(elns)>30:
            print(f"  {D}... ({len(elns)-30} more lines){X}")
        header("Demo Complete - Memory Layer Summary")
        print()
        print(f"  {G}Claw (Conversation Memory){X}")
        print("    Hot  -> WorkingMemory: recent 4 turns always injected")
        print("    Warm -> CoreMemory: LLM-compressed MEMORY.md")
        print("    Cold -> SemanticArchive: older turns embedded & vector-searched")
        print()
        print(f"  {M}Neuro Memory (Episodic & Analogy){X}")
        print("    Episodes store each Q&A as an experience capsule")
        print("    Spreading activation propagates through the memory graph")
        print("    Consolidation infers cross-episode analogous/theme relations")
        print()
        print(f"  {BL}Long-Horizon Memory{X}")
        print("    Write decisions: store / skip / update / delete")
        print("    Secret guard: blocks API keys, tokens, passwords")
        print("    Ephemeral filter: skips debug logs, temp paths")
        print("    Natural-language editing with preview & confirm")
        print("    MEMORY.md export for audit & portability")
        print()
        print(f"  {Y}AgentMemoryCore{X}")
        print("    recall()  -> unified retrieval instruction from all layers")
        print("    after_response() -> consolidate back into all layers")
        print("    Full MemoryTrace for observability")
        print()
        print(f"  {D}Workspace: {tmp}{X}")
    finally:
        try:
            shutil.rmtree(tmp)
        except:
            pass

if __name__=="__main__":
    asyncio.run(run_demo())
