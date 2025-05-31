"""
LLM上下文生成器 - 将图检索结果转换为LLM可用的上下文
"""
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import time

from langchain.schema import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.config import MODEL
# Fix: Import GraphRetriever from the right location - avoid circular import issue
from src.core.text2cypher import GraphRetriever

logger = logging.getLogger(__name__)

class LLMContextGenerator:
    """将图检索结果转换为LLM可理解的上下文"""
    
    def __init__(self, 
                graph_retriever: GraphRetriever,
                max_context_length: int = 4000,
                include_graph_visualization: bool = False):
        """初始化上下文生成器
        
        Args:
            graph_retriever: 图检索器实例
            max_context_length: 上下文最大长度
            include_graph_visualization: 是否包含图可视化描述
        """
        self.graph_retriever = graph_retriever
        self.max_context_length = max_context_length
        self.include_graph_visualization = include_graph_visualization
        logger.info("LLM上下文生成器初始化完成")
    
    def generate_context(self, query: str, retrieval_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成LLM可用的上下文
        
        Args:
            query: 用户查询
            retrieval_results: 检索结果
            
        Returns:
            包含组织好的上下文信息的字典
        """
        start_time = time.time()
        
        # 从检索结果中获取组织好的上下文
        organized_context = retrieval_results.get("organized_context", {})
        if not organized_context:
            logger.warning("检索结果中没有组织好的上下文")
            organized_context = self._organize_retrieval_results(retrieval_results)
        
        # 将结构化知识转换为文本
        context_text = self._format_context_for_llm(organized_context)
        
        # 限制上下文长度
        if len(context_text) > self.max_context_length:
            logger.info(f"上下文超出最大长度 ({len(context_text)} > {self.max_context_length})，进行裁剪")
            context_text = self._trim_context(context_text)
        
        # 构建完整的LLM上下文
        llm_context = {
            "query": query,
            "context_text": context_text,
            "organized_context": organized_context,
            "metadata": {
                "context_length": len(context_text),
                "generation_time": time.time() - start_time
            }
        }
        
        logger.info(f"为查询 '{query[:50]}...' 生成上下文，长度: {len(context_text)}")
        return llm_context
    
    def _organize_retrieval_results(self, retrieval_results: Dict[str, Any]) -> Dict[str, Any]:
        """将原始检索结果组织成结构化上下文
        
        Args:
            retrieval_results: 原始检索结果
            
        Returns:
            组织好的上下文
        """
        # 从结果中提取关键信息
        query = retrieval_results.get("query", "")
        entities = retrieval_results.get("entities", [])
        graph_results = []
        vector_results = []
        
        # 从merged_results中提取graph和vector结果
        merged_results = retrieval_results.get("merged_results", [])
        for item in merged_results:
            if item.get("source") == "graph":
                graph_results.append(item)
            else:
                vector_results.append(item)
        
        # 构建组织好的上下文
        organized_context = {
            "query_information": {
                "original_query": query,
                "identified_entities": entities,
            },
            "graph_knowledge": {
                "entities": [],
                "relationships": [],
                "paths": []
            },
            "text_context": {
                "relevant_passages": []
            },
            "combined_context": ""
        }
        
        # 处理图知识
        for item in graph_results:
            # 提取实体信息
            entity_info = {
                "id": item.get("id"),
                "label": item.get("metadata", {}).get("labels", ["Entity"])[0],
                "properties": item.get("metadata", {})
            }
            organized_context["graph_knowledge"]["entities"].append(entity_info)
            
            # 提取关系信息
            rel_types = item.get("metadata", {}).get("relationship_types", [])
            if rel_types:
                for rel_type in rel_types:
                    relation = {
                        "type": rel_type,
                        "source": item.get("id"),
                        "properties": {}
                    }
                    # 检查是否已存在相同关系
                    exists = False
                    for existing_rel in organized_context["graph_knowledge"]["relationships"]:
                        if existing_rel["type"] == relation["type"] and existing_rel["source"] == relation["source"]:
                            exists = True
                            break
                    
                    if not exists:
                        organized_context["graph_knowledge"]["relationships"].append(relation)
        
        # 处理文本上下文
        for item in vector_results:
            passage = {
                "content": item.get("content", ""),
                "source": item.get("id", "unknown"),
                "relevance": item.get("final_score", 0)
            }
            organized_context["text_context"]["relevant_passages"].append(passage)
            
        # 生成混合上下文摘要
        organized_context["combined_context"] = self._generate_combined_context(
            graph_knowledge=organized_context["graph_knowledge"],
            text_context=organized_context["text_context"]
        )
        
        return organized_context
    
    def _generate_combined_context(self, graph_knowledge: Dict, text_context: Dict) -> str:
        """生成混合上下文摘要
        
        Args:
            graph_knowledge: 图知识
            text_context: 文本上下文
            
        Returns:
            混合上下文摘要
        """
        combined_context = []
        
        # 添加图知识摘要
        if graph_knowledge.get("entities"):
            entity_summary = "# 相关实体\n"
            for entity in graph_knowledge["entities"][:5]:  # 限制数量
                entity_summary += f"- {entity.get('label', 'Entity')}: {entity.get('id', 'Unknown')}\n"
                
                # 添加实体属性
                if entity.get("properties") and isinstance(entity.get("properties"), dict):
                    properties = entity.get("properties")
                    for key, value in properties.items():
                        if key not in ["id", "embedding", "labels", "relationship_types", "distance"]:
                            entity_summary += f"  - {key}: {value}\n"
            
            combined_context.append(entity_summary)
            
        # 添加关系摘要
        if graph_knowledge.get("relationships"):
            rel_summary = "# 关系信息\n"
            for relation in graph_knowledge["relationships"][:5]:  # 限制数量
                rel_summary += f"- 类型: {relation.get('type', 'Unknown')}, 来源: {relation.get('source', 'Unknown')}\n"
            
            combined_context.append(rel_summary)
            
        # 添加路径描述
        if graph_knowledge.get("paths"):
            path = graph_knowledge["paths"][0]  # 获取第一条路径
            path_nodes = path.get("path_nodes", [])
            if path_nodes:
                path_desc = "# 实体关系路径\n"
                node_labels = [f"{node.get('labels', ['Entity'])[0]}: {node.get('id', 'Unknown')}"
                             for node in path_nodes]
                path_desc += " → ".join(node_labels) + "\n"
                combined_context.append(path_desc)
                
        # 添加文本段落
        if text_context.get("relevant_passages"):
            passage_summary = "# 相关文本片段\n"
            for i, passage in enumerate(text_context["relevant_passages"][:3]):  # 限制数量
                content = passage.get("content", "")
                # 截断过长的内容
                if len(content) > 500:
                    content = content[:497] + "..."
                passage_summary += f"## 片段 {i+1}\n{content}\n\n"
            combined_context.append(passage_summary)
            
        # 合并所有上下文
        return "\n".join(combined_context)
    
    def _format_context_for_llm(self, organized_context: Dict[str, Any]) -> str:
        """格式化上下文以供LLM使用
        
        Args:
            organized_context: 组织好的上下文
            
        Returns:
            格式化后的上下文文本
        """
        # 如果已有combined_context，直接使用
        if organized_context.get("combined_context"):
            return organized_context["combined_context"]
        
        # 否则生成新的combined_context
        return self._generate_combined_context(
            organized_context.get("graph_knowledge", {}),
            organized_context.get("text_context", {})
        )
    
    def _trim_context(self, context_text: str) -> str:
        """裁剪上下文，确保不超过最大长度
        
        Args:
            context_text: 原始上下文文本
            
        Returns:
            裁剪后的上下文文本
        """
        if len(context_text) <= self.max_context_length:
            return context_text
        
        # 分割上下文为段落
        paragraphs = context_text.split("\n\n")
        
        # 如果只有一个段落，直接截断
        if len(paragraphs) <= 1:
            return context_text[:self.max_context_length - 3] + "..."
        
        # 计算每个段落的重要性得分
        # 这里简单地用段落长度来表示重要性
        paragraph_scores = [(i, len(p), p) for i, p in enumerate(paragraphs)]
        paragraph_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 保留最重要的段落，直到达到最大长度
        selected_paragraphs = []
        current_length = 0
        
        for _, length, paragraph in sorted(paragraph_scores, key=lambda x: x[0]):
            if current_length + length + 2 <= self.max_context_length:  # +2 for newlines
                selected_paragraphs.append(paragraph)
                current_length += length + 2
            else:
                # 如果该段落不能完全加入，尝试截断它
                remaining_length = self.max_context_length - current_length - 3  # -3 for ellipsis
                if remaining_length > 50:  # 只有当剩余空间足够时才截断
                    truncated = paragraph[:remaining_length] + "..."
                    selected_paragraphs.append(truncated)
                break
        
        # 按原始顺序重新排列段落
        selected_indices = [paragraph_scores.index((i, len(p), p)) for i, p in enumerate(selected_paragraphs)]
        selected_paragraphs = [selected_paragraphs[i] for i in sorted(selected_indices)]
        
        return "\n\n".join(selected_paragraphs)
    
    def create_chat_messages(self, llm_context: Dict[str, Any], query: str, 
                            system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """创建聊天消息，包括系统消息、用户消息等
        
        Args:
            llm_context: LLM上下文
            query: 用户查询
            system_prompt: 系统提示词
            
        Returns:
            用于聊天模型的消息列表
        """
        messages = []
        
        # 添加系统消息
        if not system_prompt:
            system_prompt = (
                "你是一个基于知识图谱增强的AI助手。使用提供的上下文信息回答用户问题。"
                "上下文包含从知识图谱和文本源中检索的信息。"
                "如果上下文中没有足够信息，请说明你无法回答，而不是编造答案。"
                "回答中可以引用实体关系和文本片段，但要自然地融入回答，不要出现'根据上下文'之类的表述。"
            )
        
        # 添加系统消息
        messages.append({
            "role": "system",
            "content": system_prompt
        })
        
        # 添加包含上下文的用户消息
        context_text = llm_context.get("context_text", "")
        user_message = f"以下是与我的问题相关的上下文信息:\n\n{context_text}\n\n我的问题是: {query}"
        
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        return messages
        
    def generate_response(self, llm, llm_context: Dict[str, Any], query: str,
                         system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """生成LLM响应
        
        Args:
            llm: 语言模型
            llm_context: LLM上下文
            query: 用户查询
            system_prompt: 系统提示词
            
        Returns:
            包含响应内容的字典
        """
        start_time = time.time()
        
        # 创建聊天消息
        messages = self.create_chat_messages(llm_context, query, system_prompt)
        
        # 调用LLM生成响应
        try:
            # 调用方式取决于LLM的类型
            if hasattr(llm, "invoke"):
                # LangChain风格的调用
                response = llm.invoke(messages)
                
                # 提取响应内容，具体方式取决于返回类型
                if hasattr(response, "content"):
                    response_text = response.content
                elif isinstance(response, dict) and "content" in response:
                    response_text = response["content"]
                else:
                    response_text = str(response)
            else:
                # 直接调用（例如原生OpenAI API）
                response = llm(messages)
                response_text = response.choices[0].message.content
                
            # 构建响应字典
            result = {
                "query": query,
                "response": response_text,
                "elapsed_time": time.time() - start_time,
                "context_used": True,
                "context_length": len(llm_context.get("context_text", "")),
            }
            
            logger.info(f"为查询 '{query[:50]}...' 生成响应，耗时: {result['elapsed_time']:.2f}秒")
            return result
            
        except Exception as e:
            logger.error(f"生成响应时出错: {str(e)}")
            return {
                "query": query,
                "response": f"生成响应时出错: {str(e)}",
                "elapsed_time": time.time() - start_time,
                "context_used": False,
                "error": str(e)
            }