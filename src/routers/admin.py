from fastapi import logger
from src import app
from src.services.graphrag_service import get_graphrag_service, GraphRAGService
from src.core.text2cypher_manager import Text2CypherIndexManager
import traceback

@app.post("/api/optimize-indexes")
async def optimize_indexes():
    """优化索引以提高检索质量"""
    try:
        # 获取GraphRAG服务实例
        service = get_graphrag_service()
        
        # 执行索引优化
        success = service._optimize_text2cypher_indexes()
        
        if success:
            return {"status": "success", "message": "索引优化完成"}
        else:
            return {"status": "error", "message": "索引优化失败，请查看日志"}
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"索引优化失败: {str(e)}\n{error_traceback}")
        return {"status": "error", "message": f"索引优化失败: {str(e)}"}