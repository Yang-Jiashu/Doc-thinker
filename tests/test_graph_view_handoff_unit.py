"""Offline URL/state handoff checks; no app startup, camera, or model loading."""

import ast
import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from flask import Flask, jsonify, request
from jinja2 import Environment, FileSystemLoader


ROOT = Path(__file__).parents[1]
UI = ROOT / "docthinker/ui"


@pytest.fixture
def classic_html():
    # Render only Jinja. Importing the production Flask app would load configuration.
    env = Environment(loader=FileSystemLoader(UI / "templates"))
    env.globals["url_for"] = lambda endpoint, **values: (
        "/" + endpoint + "/" + values.get("filename", "")
    )
    return env.get_template("kg_viz_modern.html").render(
        request={"endpoint": "knowledge_graph_page"},
        api_config={"api_prefix": "/api/v1"},
        config={},
    )


def _function(html, name):
    match = re.search(
        rf"^(?:async )?function {re.escape(name)}\([^\n]*\)[^\n]*"
        r"[\s\S]*?(?=^(?:async )?function |\Z)",
        html,
        re.MULTILINE,
    )
    assert match, f"Missing classic graph function: {name}"
    return match.group(0)


def _run_js(html, names, setup, checks, *, extra=""):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for offline graph behavior checks")
    script = "(async () => {\n" + setup + "\n"
    script += "\n".join(_function(html, name) for name in names)
    script += "\n" + extra + "\n" + checks + "\n})()"
    result = subprocess.run(
        [node, "-e", (
            "const fs=require('fs'),vm=require('vm'),assert=require('node:assert/strict');"
            "const script=JSON.parse(fs.readFileSync(0,'utf8'));"
            "Promise.resolve(new vm.Script(script).runInNewContext({assert,URL,URLSearchParams,console}))"
            ".catch(error=>{console.error(error);process.exitCode=1});"
        )],
        input=json.dumps(script), text=True, capture_output=True, timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("explicit", ["#00003", "不存在 / #?&"])
def test_classic_url_session_takes_priority_and_invalid_session_never_falls_back(classic_html, explicit):
    _run_js(classic_html, ["ensureKgSessionReady"], """
        let currentSessionId = 'previous', currentGraphView = 'semantic';
        let pendingGraphNode = '', graphLayoutMode = 'semantic';
        const writes = [], messages = [], selections = [];
        const selector = { value: '' };
        const document = { getElementById: () => selector };
        const window = { location: { search: '?' + new URLSearchParams({
            session_id: EXPLICIT, node: '能源 # / & ?', search: '风险 + 成本'
        }) } };
        async function fetchSessions() { return [{id:'#00001'},{id:'#00003'}]; }
        function getStoredSessionId() { return '#00001'; }
        function setStoredSessionId(value) { writes.push(value); }
        function readStandardGraphState(sid) { return {node:'old',view:'semantic'}; }
        function renderSessionSelector(sessions, selected) { selections.push(selected); selector.value = selected || '#00001'; }
        function showToast(message) { messages.push(message); }
    """.replace("EXPLICIT", json.dumps(explicit)), """
        await ensureKgSessionReady();
        if (window.location.search.includes('%2300003')) {
            assert.equal(currentSessionId, '#00003');
            assert.equal(pendingGraphNode, '能源 # / & ?');
            assert.equal(writes.join(','), '#00003');
            assert.equal(messages.length, 0);
        } else {
            assert.equal(currentSessionId, '');
            assert.equal(selector.value, '');
            assert.equal(writes.length, 0);
            assert.equal(messages.length, 1);
        }
    """)


@pytest.mark.parametrize(
    "node,return_view,expected",
    [("能源 # / & ?", None, "semantic"), ("能源", "causal", "causal"), (None, None, "causal")],
)
def test_explicit_node_link_overrides_old_causal_snapshot_unless_view_is_explicit(classic_html, node, return_view, expected):
    params = {"session_id": "#00003"}
    if node is not None:
        params["node"] = node
    if return_view is not None:
        params["return_view"] = return_view
    _run_js(classic_html, ["ensureKgSessionReady", "loadRealData"], """
        let currentSessionId = '', currentGraphView = 'semantic';
        let pendingGraphNode = '', graphLayoutMode = 'semantic', graphLoadRevision = 0;
        let graphData = {nodes:[],links:[]};
        const calls = [], hint = {textContent:''};
        const document = {getElementById:()=>hint};
        const params = new URLSearchParams(PARAMS);
        const window = {location:{search:'?' + params}};
        async function fetchSessions() { return [{id:'#00003'}]; }
        function getStoredSessionId() { return '#00003'; }
        function setStoredSessionId() {}
        function renderSessionSelector() {}
        function readStandardGraphState() { return {view:'causal',node:'stale-causal-node'}; }
        function showToast() { throw new Error('Unexpected warning'); }
        function getKgApiPrefix() { return '/api/v1'; }
        function renderGraph() {}
        async function fetch(url) {
            calls.push(url);
            return {ok:true,text:async()=>JSON.stringify({nodes:[],edges:[]})};
        }
    """.replace("PARAMS", json.dumps(params)), """
        await ensureKgSessionReady();
        assert.equal(currentGraphView, EXPECTED);
        await loadRealData();
        const url = new URL(calls[0], 'https://local.invalid');
        if (currentGraphView === 'semantic') {
            assert.equal(url.pathname, '/api/v1/knowledge-graph/data');
            assert.equal(url.searchParams.get('focus_node'), params.get('node'));
        } else {
            assert.equal(url.pathname, '/api/v1/knowledge-graph/causal-data');
        }
    """.replace("EXPECTED", json.dumps(expected)))


@pytest.mark.parametrize("view", ["semantic", "causal"])
def test_open_immersive_encodes_view_state_and_stops_camera_before_navigation(classic_html, view):
    _run_js(classic_html, ["openPromoGraph"], """
        const calls = [];
        const currentSessionId = '#00003';
        const selectedNode = {id:'能源 / #?&= + https://elsewhere.invalid'};
        const currentGraphView = VIEW;
        const document = {getElementById: () => ({value:'风险 ?成本&收益 + /'})};
        function saveStandardGraphState() { calls.push('save'); }
        let destination;
        const window = {
            KGSharedGesture: {stop: () => calls.push('stop')},
            location: {assign: value => { calls.push('navigate'); destination = value; }}
        };
    """.replace("VIEW", json.dumps(view)), """
        openPromoGraph();
        assert.equal(calls.join(','), 'save,stop,navigate');
        const url = new URL(destination, 'https://local.invalid');
        assert.equal(url.origin, 'https://local.invalid');
        assert.equal(url.pathname, '/promo-graph');
        assert.equal(url.searchParams.get('session_id'), '#00003');
        assert.equal(url.searchParams.get('node'), currentGraphView === 'semantic' ? selectedNode.id : '');
        assert.equal(url.searchParams.get('search'), '风险 ?成本&收益 + /');
        assert.equal(url.searchParams.get('return_view'), currentGraphView);
        assert.equal(url.hash, '');
        assert.equal(url.searchParams.has('gesture'), false);
    """)


def test_restore_requests_bounded_focus_then_selects_exact_node_and_restores_search(classic_html):
    _run_js(classic_html, ["loadRealData", "restoreGraphHandoff"], """
        let currentSessionId = '#00003', currentGraphView = 'semantic';
        let graphLoadRevision = 0, graphData = {nodes:[],links:[]};
        let pendingGraphNode = '能源 # / & ?';
        const calls = [], selected = [], searched = [], warnings = [];
        const elements = {'search-input':{value:''},'data-source-hint':{textContent:''}};
        const document = {getElementById: id => elements[id]};
        const window = {location: {search:'?' + new URLSearchParams({
            session_id:currentSessionId,node:pendingGraphNode,search:'进口 + 风险'
        })}};
        function getKgApiPrefix() { return '/api/v1'; }
        function renderGraph() {}
        function selectNode(value) { selected.push(value.id); }
        function searchNodes(value) { searched.push(value); }
        function showToast(message) { warnings.push(message); }
        function readStandardGraphState() { return {node:'different',search:'stale'}; }
        async function fetch(url) {
            calls.push(url);
            return {ok:true,text:async()=>JSON.stringify({
                nodes:[{id:pendingGraphNode,label:'能源',type:'concept'}],edges:[],
                metadata:{nodes_returned:1,total_nodes:260}
            })};
        }
    """, """
        await loadRealData();
        restoreGraphHandoff();
        const url = new URL(calls[0], 'https://local.invalid');
        assert.equal(url.searchParams.get('session_id'), '#00003');
        assert.equal(url.searchParams.get('focus_node'), pendingGraphNode);
        assert.notEqual(url.searchParams.get('scope'), 'full');
        assert.equal(url.hash, '');
        assert.equal(selected.join(','), pendingGraphNode);
        assert.equal(elements['search-input'].value, '进口 + 风险');
        assert.equal(searched.length, 0);
        assert.equal(warnings.length, 0);
        graphData = {nodes:[],links:[]};
        restoreGraphHandoff();
        assert.equal(warnings.length, 1);
        assert.equal(searched[0], '进口 + 风险');
    """)


def test_explicit_empty_handoff_values_do_not_revive_saved_selection_or_search(classic_html):
    _run_js(classic_html, ["ensureKgSessionReady", "loadRealData", "restoreGraphHandoff"], """
        let currentSessionId = '', currentGraphView = 'semantic', graphLayoutMode = 'semantic';
        let graphLoadRevision = 0, pendingGraphNode = '', graphData = {nodes:[],links:[]};
        const calls = [], elements = {'search-input':{value:'previous'},'data-source-hint':{textContent:''}};
        const document = {getElementById:id=>elements[id]};
        const window = {location:{search:'?session_id=%2300003&node=&search='}};
        async function fetchSessions() { return [{id:'#00003'}]; }
        function getStoredSessionId() { return '#00003'; }
        function setStoredSessionId() {}
        function renderSessionSelector() {}
        function readStandardGraphState() { return {node:'old-node',search:'old search',view:'causal'}; }
        function showToast() { throw new Error('Cleared selection should not warn'); }
        function selectNode() { throw new Error('Cleared selection was restored'); }
        function searchNodes() { throw new Error('Cleared search was restored'); }
        function getKgApiPrefix() { return '/api/v1'; }
        function renderGraph() {}
        async function fetch(url) {
            calls.push(url);
            return {ok:true,text:async()=>JSON.stringify({nodes:[{id:'old-node'}],edges:[]})};
        }
    """, """
        await ensureKgSessionReady();
        assert.equal(currentGraphView, 'semantic');
        assert.equal(pendingGraphNode, '');
        await loadRealData();
        assert.equal(new URL(calls[0], 'https://local.invalid').searchParams.has('focus_node'), false);
        restoreGraphHandoff();
        assert.equal(elements['search-input'].value, '');
    """)


def test_classic_fifty_node_projection_keeps_handoff_target_without_expanding_cap(classic_html):
    # Run the actual filtering/capping prefix, excluding D3 rendering effects.
    render_prefix = _function(classic_html, "renderGraph").split("    const container =", 1)[0]
    assert "visibleNodeIds" in render_prefix
    bounded_render = render_prefix + "return {nodes:visibleNodes,links:visibleLinks};\n}"
    _run_js(classic_html, ["isExpandedNode", "nodeDegree", "edgeSourceId", "edgeTargetId"], """
        const window = {}, g = {selectAll:()=>({remove(){}})};
        let simulation = null, focusedNodeId = null, selectedNode = null;
        const MAX_RENDER_NODES = 50, visibleLevels = {1:true,2:true,3:true,expanded:true};
        const pendingGraphNode = 'low-degree-target';
        const graphData = {nodes:Array.from({length:200},(_,i)=>({
            id:i===199?pendingGraphNode:String(i),level:2,degree:200-i
        })),links:[{source:'0',target:pendingGraphNode},{source:'0',target:'150'}]};
    """, """
        const result = renderGraph();
        assert.equal(result.nodes.length, 50);
        assert.equal(result.nodes.some(item=>item.id===pendingGraphNode), true);
        assert.equal(result.links.length, 1);
        assert.equal(result.links[0].target, pendingGraphNode);
        assert.equal(graphData.nodes.length, 200);
    """, extra=bounded_render)


def test_classic_obsolete_graph_response_cannot_overwrite_new_session(classic_html):
    _run_js(classic_html, ["loadRealData"], """
        let currentSessionId = '#one', currentGraphView = 'semantic';
        let graphLoadRevision = 0, pendingGraphNode = '';
        let graphData = {nodes:[],links:[]};
        const hint = {textContent:''}, pending = [];
        const document = {getElementById:()=>hint};
        function getKgApiPrefix() { return '/api/v1'; }
        function renderGraph() {}
        async function fetch(url) { return {ok:true,text:()=>new Promise(resolve=>pending.push(resolve))}; }
    """, """
        const oldLoad = loadRealData();
        await Promise.resolve();
        currentSessionId = '#two';
        const newLoad = loadRealData();
        await Promise.resolve();
        pending[1](JSON.stringify({nodes:[{id:'shared',description:'new session evidence'}],edges:[]}));
        await newLoad;
        pending[0](JSON.stringify({nodes:[{id:'shared',description:'old session evidence'}],edges:[]}));
        await oldLoad;
        assert.equal(graphData.nodes[0].description, 'new session evidence');
        assert.equal(hint.textContent.includes('#two'), true);
    """)


def test_classic_legacy_gesture_url_only_opens_settings_and_never_starts_camera(classic_html):
    onload = re.search(r"window\.onload = async \(\) => \{[\s\S]*?\n\};", classic_html)
    assert onload
    _run_js(classic_html, [], """
        const calls = [];
        const window = {location:{search:'?session_id=%2300003&gesture=classic'},
            openGestureExperienceChooser:()=>calls.push('settings')};
        async function initGraph() { calls.push('graph'); }
        function restoreGraphHandoff() { calls.push('restore'); }
        function startGestureControl() { throw new Error('Implicit camera activation'); }
    """, """
        await window.onload();
        assert.equal(calls.join(','), 'graph,restore,settings');
        assert.equal(window.location.search, '?session_id=%2300003&gesture=classic');
    """, extra=onload.group(0))
    adapter = (UI / "promo_graph/static/gesture_chooser.js").read_text()
    assert "runtime.GestureController" in adapter
    assert "@mediapipe/tasks-vision@latest" not in adapter
    # Viewing settings imports the shared module, but does not call start().
    chooser = adapter.split("window.openGestureExperienceChooser =", 1)[1].split("\n    };", 1)[0]
    assert ".start(" not in chooser


def test_classic_gesture_release_restores_only_captured_nodes_original_constraints():
    adapter = (UI / "promo_graph/static/gesture_chooser.js").read_text()
    callbacks = re.search(
        r"                onNodeAttractionStart\(index\) \{[\s\S]*?(?=                onStateChange:)",
        adapter,
    )
    assert callbacks, "Missing shared gesture attraction callbacks"
    _run_js("", [], """
        let attractedNode = null, stops = 0, resumes = 0;
        const original = {id:'dragged',x:10,y:20,fx:0,fy:42};
        const unrelated = {id:'pinned',x:30,y:40,fx:900,fy:901};
        const free = {id:'free',x:50,y:60,fx:null,fy:undefined};
        let graphData = {nodes:[original,unrelated,free]};
        const item = index => graphData.nodes[index];
        const simulation = {stop:()=>stops++};
        function resumeRenderedGraphForceLayout() { resumes++; }
    """, """
        callbacks.onNodeAttractionStart(0);
        assert.equal(original.fx, 10);
        assert.equal(original.fy, 20);
        original.fx = 100; original.fy = 200;
        callbacks.onNodeAttractionEnd();
        assert.equal(original.fx, 0);
        assert.equal(original.fy, 42);
        assert.equal(unrelated.fx, 900);
        assert.equal(unrelated.fy, 901);
        assert.equal(free.fx, null);
        assert.equal(free.fy, undefined);
        assert.equal(resumes, 1);
        callbacks.onNodeAttractionEnd();
        assert.equal(resumes, 1);

        callbacks.onNodeAttractionStart(2);
        assert.equal(free.fx, 50);
        // A graph replacement must not redirect cleanup onto new session nodes.
        graphData = {nodes:[{id:'free',fx:999,fy:998}]};
        callbacks.onNodeAttractionEnd();
        assert.equal(free.fx, null);
        assert.equal(free.fy, undefined);
        assert.equal(graphData.nodes[0].fx, 999);
        assert.equal(graphData.nodes[0].fy, 998);
        assert.equal(stops, 2);
        assert.equal(resumes, 2);
        assert.equal(attractedNode, null);
    """, extra="const callbacks = {\n" + callbacks.group(0) + "\n};")


def test_classic_mouse_takeover_stays_paused_and_error_message_has_priority():
    adapter = (UI / "promo_graph/static/gesture_chooser.js").read_text()
    # Run the real adapter body in our VM scope, injecting only its controller.
    # The module loader remains uncalled: there is no model/camera initialization.
    body = adapter.split("(function () {", 1)[1].rsplit("})();", 1)[0]
    _run_js("", [], """
        const events = new Map(), windowEvents = new Map(), timers = [], changes = [];
        const inside = {}, outside = {}, controls = new Map();
        const elements = new Map([
            ['graph-container',{contains:target=>target===inside}],
            ['shared-gesture-state',{textContent:''}], ['gesture-mode',{textContent:''}],
            ['gesture-layer',{classList:{toggle(){}}}]
        ]);
        const document = {
            currentScript:{dataset:{}}, hidden:false,
            createElement:()=>({}), head:{append(){}},
            getElementById:id=>elements.get(id),
            addEventListener:(name,callback)=>{
                if (!events.has(name)) events.set(name,[]);
                events.get(name).push(callback);
            }
        };
        const window = {addEventListener:(name,callback)=>windowEvents.set(name,callback)};
        const state = {enabled:true,suspended:false,starting:false,error:''};
        const mockController = {
            getState:()=>state,
            setSuspended:value=>{ changes.push(value); state.suspended=value; }
        };
        function setTimeout(callback) { timers.push(callback); return timers.length; }
        function clearTimeout() {}
        function fire(name,event={}) { for (const callback of events.get(name)||[]) callback(event); }
    """, """
        controller = mockController;
        fire('pointerdown',{target:outside});
        assert.equal(changes.length, 0);
        fire('pointerdown',{target:inside});
        assert.equal(changes.join(','), 'true');
        fire('pointerup',{target:inside});
        fire('pointercancel',{target:inside});
        while (timers.length) timers.shift()();
        document.hidden = false; fire('visibilitychange');
        assert.equal(changes.includes(false), false);
        assert.equal(state.suspended, true);

        dialog = {querySelector:selector=>{
            if (!controls.has(selector)) controls.set(selector,{textContent:'',disabled:false});
            return controls.get(selector);
        }};
        state.error = '手势识别暂停：帧处理失败';
        updateStatus();
        assert.equal(elements.get('shared-gesture-state').textContent, state.error);
        assert.equal(elements.get('gesture-mode').textContent, state.error);
        assert.equal(controls.get('#classic-gesture-status').textContent, state.error);
    """, extra=textwrap.dedent(body))


def test_flask_proxy_preserves_focus_and_encoded_session_without_starting_app():
    # Execute just the production handler against a local Flask request context.
    # This avoids config/.env loading and still verifies real requests kwargs.
    tree = ast.parse((UI / "app.py").read_text())
    handler = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "kg_data_proxy")
    handler.decorator_list = []
    namespace = {"request": request, "jsonify": jsonify, "api_config": SimpleNamespace(api_prefix="/api/v1")}
    exec(compile(ast.Module(body=[handler], type_ignores=[]), str(UI / "app.py"), "exec"), namespace)
    app = Flask(__name__)
    payload = {"nodes": [{"id": "能源 # / ?&"}], "edges": [], "metadata": {"focus_node_found": True}}
    response = Mock(status_code=200)
    response.json.return_value = payload
    with app.test_request_context("/api/v1/knowledge-graph/data", query_string={
        "session_id": "#00003", "focus_node": "能源 # / ?&",
    }), patch("requests.get", return_value=response) as get:
        result, status = namespace["kg_data_proxy"]()
    assert status == 200
    assert result.get_json() == payload
    assert get.call_args.kwargs["params"] == {
        "session_id": "#00003", "scope": "summary", "focus_node": "能源 # / ?&",
    }
    assert get.call_args.kwargs["timeout"] == 30
