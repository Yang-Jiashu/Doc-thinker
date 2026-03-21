#标记文档在批处理流水线中的状态。
from enum import Enum


class DocStatus(str, Enum):
    """Document processing status"""

    READY = "ready"
    HANDLING = "handling"
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
