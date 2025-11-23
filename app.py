# app.py
import json
from typing import List, Dict

from flask import Flask, request, Response, jsonify, send_from_directory

from agent import Agent, AgentConfig

app = Flask(__name__, static_folder="static", static_url_path="/static")


@app.route("/")
def index():
    # 直接返回 static/index.html
    return send_from_directory(app.static_folder, "index.html")


def build_agents_from_request(agents_data: List[Dict]) -> List[Agent]:
    agents: List[Agent] = []
    for a in agents_data:
        name = a.get("name", "").strip()
        if not name:
            continue

        cfg = AgentConfig(
            name=name,
            short_name=name,
            role_description=a.get("role", ""),
            strategic_preferences=a.get("strategic_preferences", ""),
            knowledge=a.get("knowledge", ""),
            knowledge_mode=a.get("knowledge_mode", "inline"),
            knowledge_folder=a.get("knowledge_folder", ""),
        )
        agents.append(Agent(cfg))
    return agents


@app.route("/simulate_stream", methods=["POST"])
def simulate_stream():
    """
    推演流式接口：
    - 前端 POST JSON: {scenario, turns, agents:[{name, role, knowledge, knowledge_mode, knowledge_folder}]}
    - 后端逐轮推演，每生成一个智能体的发言，就立刻通过 chunk 方式写回前端
    - 约定：每一行是一条 JSON，格式 {"type":"message","data":{turn,speaker,content}}
    """
    data = request.get_json(force=True)
    scenario = data.get("scenario", "").strip()
    turns = int(data.get("turns", 3))
    agents_data = data.get("agents", [])

    agents = build_agents_from_request(agents_data)

    if not agents:
        return jsonify({"error": "No valid agents provided."}), 400

    def generate():
        messages_log = []
        # 初始全局局势
        global_context = f"场景描述：{scenario}" if scenario else "场景描述：无（前端未提供）"

        try:
            for t in range(turns):
                turn_idx = t + 1
                for agent in agents:
                    reply = agent.act(global_context)

                    msg = {
                        "turn": turn_idx,
                        "speaker": agent.config.name,
                        "content": reply,
                    }
                    messages_log.append(msg)

                    # 更新全局局势（简单做法：把每轮发言直接附加进去）
                    global_context += f"\n\n[第{turn_idx}轮 {agent.config.name} 发言]: {reply}"

                    # 一条消息一行 JSON
                    chunk = json.dumps({"type": "message", "data": msg}, ensure_ascii=False) + "\n"
                    yield chunk

            # 最后发一个 done 事件
            done_chunk = json.dumps(
                {"type": "done", "data": {"total_messages": len(messages_log)}},
                ensure_ascii=False
            ) + "\n"
            yield done_chunk

        except Exception as e:
            err_chunk = json.dumps(
                {"type": "error", "data": f"{repr(e)}"},
                ensure_ascii=False
            ) + "\n"
            yield err_chunk

    return Response(generate(), mimetype="text/plain; charset=utf-8")


@app.route("/simulate", methods=["POST"])
def simulate_once():
    """
    非流式版本：一次性返回所有推演结果（保留，方便调试或脚本调用）。
    """
    data = request.get_json(force=True)
    scenario = data.get("scenario", "").strip()
    turns = int(data.get("turns", 3))
    agents_data = data.get("agents", [])

    agents = build_agents_from_request(agents_data)
    if not agents:
        return jsonify({"error": "No valid agents provided."}), 400

    messages_log = []
    global_context = f"场景描述：{scenario}" if scenario else "场景描述：无（前端未提供）"

    for t in range(turns):
        turn_idx = t + 1
        for agent in agents:
            reply = agent.act(global_context)
            msg = {
                "turn": turn_idx,
                "speaker": agent.config.name,
                "content": reply,
            }
            messages_log.append(msg)
            global_context += f"\n\n[第{turn_idx}轮 {agent.config.name} 发言]: {reply}"

    return jsonify({"messages": messages_log})


if __name__ == "__main__":
    # Flask 启动
    # 生产环境可以用 gunicorn 等 WSGI 容器
    app.run(host="0.0.0.0", port=8000, debug=True)
