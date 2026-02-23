"""
Session Manager - 会话管理

管理多用户/多会话状态
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Session:
    """会话"""
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)


class SessionManager:
    """会话管理器"""
    
    def __init__(self, max_history: int = 100):
        self.sessions: Dict[str, Session] = {}
        self.max_history = max_history
    
    def create_session(self, metadata: Optional[Dict] = None) -> str:
        """创建新会话"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = Session(
            session_id=session_id,
            metadata=metadata or {},
        )
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话"""
        session = self.sessions.get(session_id)
        if session:
            session.last_active = time.time()
        return session
    
    def add_message(self, session_id: str, role: str, content: str):
        """添加消息到会话历史"""
        session = self.get_session(session_id)
        if not session:
            return
        
        session.history.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })
        
        # 限制历史长度
        if len(session.history) > self.max_history:
            session.history = session.history[-self.max_history:]
    
    def get_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """获取会话历史"""
        session = self.get_session(session_id)
        if not session:
            return []
        return session.history[-limit:]
    
    def delete_session(self, session_id: str):
        """删除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def list_sessions(self) -> List[str]:
        """列出所有会话ID"""
        return list(self.sessions.keys())
    
    def cleanup_inactive(self, max_age_hours: float = 24):
        """清理不活跃会话"""
        now = time.time()
        to_delete = []
        for session_id, session in self.sessions.items():
            if (now - session.last_active) > (max_age_hours * 3600):
                to_delete.append(session_id)
        
        for session_id in to_delete:
            self.delete_session(session_id)
        
        return len(to_delete)
