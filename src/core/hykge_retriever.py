import logging
import json
import numpy as np
from typing import List, Dict, Any, Optional, Union
from src.core.graph_retrieval import GraphRetriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class HyKGERetriever:
    """基于HyKGE算法的知识图谱增强检索类"""
    
    def __init__(self, llm, embeddings, graph_retriever: GraphRetriever = None,
                 ner_model=None, top_k: int = 5, max_hop: int = 2):
        """初始化HyKGE检索器
        
        Args:
            graph_retriever: GraphRetriever对象
            llm: 大语言模型
            embeddings: 嵌入模型
            ner_model: 命名实体识别模型（如果为None，使用LLM进行实体提取）
            top_k: 重排后保留的路径数量
            max_hop: 最大跳数（实体间关系路径最大长度）
        """
        self.llm = llm
        self.embeddings = embeddings
        self.ner_model = ner_model
        self.top_k = top_k
        self.max_hop = max_hop
        if not graph_retriever:
            self.graph_retriever = GraphRetriever(uri="your_neo4j_uri", user="your_neo4j_user", password="your_neo4j_password", llm=llm, ner_model=ner_model)
        else:
            self.graph_retriever = graph_retriever
        logger.info("HyKGE检索器初始化完成")
    
    def pre_embed_entities(self):
        """预嵌入知识图谱中的实体（算法步骤1-3）"""
        logger.info("开始预嵌入知识图谱实体")
        cypher_query = """
        MATCH (e:__Entity__)
        WHERE e.embedding IS NULL
        RETURN e.id AS id, e.name AS name
        LIMIT 1000
        """
        
        with self.graph_retriever.driver.session(database=self.graph_retriever.database) as session:
            result = session.run(cypher_query)
            entities = [(record["id"], record["name"]) for record in result]
            
        # 批量嵌入实体名称
        entity_names = [entity[1] for entity in entities if entity[1]]
        entity_ids = [entity[0] for entity in entities if entity[1]]
        
        if not entity_names:
            logger.info("没有需要嵌入的新实体")
            return 0
            
        # 分批处理以避免内存问题
        batch_size = 100
        total_embedded = 0
        
        for i in range(0, len(entity_names), batch_size):
            batch = entity_names[i:i+batch_size]
            batch_ids = entity_ids[i:i+batch_size]
            try:
                embeddings = self.embeddings.embed_documents(batch)
                
                # 更新Neo4j中的实体嵌入
                for j, embedding in enumerate(embeddings):
                    entity_id = batch_ids[j]
                    self._update_entity_embedding(entity_id, embedding)
                    total_embedded += 1
                
                logger.info(f"批量嵌入{len(batch)}个实体完成")
            except Exception as e:
                logger.error(f"批量嵌入实体失败: {str(e)}")
        
        return total_embedded
    
    def _update_entity_embedding(self, entity_id: str, embedding: List[float]):
        """更新实体的嵌入向量"""
        cypher_query = """
        MATCH (e:__Entity__ {id: $entity_id})
        SET e.embedding = $embedding
        """
        
        with self.graph_retriever.driver.session(database=self.graph_retriever.database) as session:
            session.run(cypher_query, entity_id=entity_id, embedding=embedding)
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """从文本中提取实体（算法步骤5）
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表 [{"entity": "实体名", "type": "实体类型"}]
        """
        logger.info(f"从文本中提取实体: {text[:100]}...")
        
        if self.ner_model:
            # 使用专用NER模型提取实体
            entities = self.ner_model(text)
            return entities
        else:
            # 使用LLM提取实体
            prompt = f"""
            从以下文本中提取所有的命名实体（人物、组织、地点、产品等）:
            
            {text}
            
            以JSON数组格式返回，每个实体包含实体名称和实体类型:
            [
                {{"entity": "实体名称", "type": "实体类型"}}
            ]
            仅返回JSON数组，不要包含其他解释。
            """
            
            try:
                result = self.llm.invoke(prompt)
                
                # 处理LLM返回的JSON
                import re
                json_pattern = r'\[(.*?)\]'
                match = re.search(json_pattern, result, re.DOTALL)
                
                if match:
                    try:
                        json_str = f"[{match.group(1)}]"
                        entities = json.loads(json_str)
                        return entities
                    except json.JSONDecodeError:
                        try:
                            # 尝试直接解析整个结果
                            entities = json.loads(result)
                            if isinstance(entities, list):
                                return entities
                        except:
                            logger.warning(f"无法解析LLM返回的实体JSON: {result}")
                            return []
                else:
                    logger.warning("LLM未返回有效的实体JSON")
                    return []
            except Exception as e:
                logger.error(f"使用LLM提取实体失败: {str(e)}")
                return []
    
    def entity_linking(self, entities: List[Dict[str, str]]) -> List[str]:
        """将提取的实体链接到知识图谱（算法步骤6）
        
        Args:
            entities: 从文本中提取的实体列表
            
        Returns:
            链接到知识图谱的实体ID列表
        """
        linked_entities = []
        
        for entity in entities:
            entity_name = entity["entity"] if isinstance(entity, dict) else entity
            
            # 使用全文索引搜索实体
            cypher_query = """
            CALL db.index.fulltext.queryNodes("fulltext_entity_id", $query_text, {limit: 5}) 
            YIELD node, score
            WHERE score > 0.5 AND 'Entity' IN labels(node)
            RETURN node.id AS id, score
            ORDER BY score DESC
            LIMIT 1
            """
            
            with self.graph_retriever.driver.session(database=self.graph_retriever.database) as session:
                try:
                    result = session.run(cypher_query, query_text=entity_name)
                    record = result.single()
                    
                    if record:
                        linked_entities.append(record["id"])
                        logger.debug(f"实体'{entity_name}'链接到KG中的'{record['id']}'")
                except Exception as e:
                    logger.error(f"实体链接查询失败: {str(e)}")
        
        logger.info(f"共链接了{len(linked_entities)}个实体到知识图谱")
        return linked_entities
    
    def retrieve_reasoning_chains(self, entity_ids: List[str]) -> List[Dict[str, Any]]:
        """检索实体间的推理链（算法步骤7）
        
        Args:
            entity_ids: 实体ID列表
            
        Returns:
            实体间的推理链
        """
        reasoning_chains = []
        
        # 获取所有实体对之间的路径
        for i in range(len(entity_ids)):
            for j in range(i+1, len(entity_ids)):
                source_id = entity_ids[i]
                target_id = entity_ids[j]
                
                logger.debug(f"检索实体'{source_id}'和'{target_id}'之间的推理链")
                
                cypher_query = """
                MATCH path = (source:__Entity__ {id: $source_id})-[*1..$max_hop]-(target:__Entity__ {id: $target_id})
                WHERE ALL(rel IN relationships(path) WHERE type(rel) <> 'MENTIONED_IN')
                WITH path, length(path) as path_length
                ORDER BY path_length
                LIMIT 5
                RETURN path, path_length
                """
                
                with self.graph_retriever.driver.session(database=self.graph_retriever.database) as session:
                    try:
                        result = session.run(cypher_query, source_id=source_id, target_id=target_id, max_hop=self.max_hop)
                        
                        for record in result:
                            path = record["path"]
                            nodes = []
                            rels = []
                            
                            for node in path.nodes:
                                node_props = dict(node._properties)
                                node_labels = list(node.labels)
                                nodes.append({
                                    "id": node_props.get("id", str(node.id)),
                                    "name": node_props.get("name", ""),
                                    "type": node_labels[0] if node_labels else "Unknown"
                                })
                            
                            for rel in path.relationships:
                                rels.append({
                                    "type": rel.type,
                                    "properties": dict(rel._properties)
                                })
                            
                            chain = {
                                "source": source_id,
                                "target": target_id,
                                "nodes": nodes,
                                "relationships": rels,
                                "length": record["path_length"]
                            }
                            
                            reasoning_chains.append(chain)
                    except Exception as e:
                        logger.error(f"检索推理链失败: {str(e)}")
        
        logger.info(f"共检索到{len(reasoning_chains)}条推理链")
        return reasoning_chains
    
    def rerank_reasoning_chains(self, reasoning_chains: List[Dict], hypothesis_output: str) -> List[Dict]:
        """基于假设输出重排推理链（算法步骤8）
        
        Args:
            reasoning_chains: 推理链列表
            hypothesis_output: LLM的假设输出
            
        Returns:
            重排后的推理链
        """
        if not reasoning_chains:
            return []
        
        logger.info(f"对{len(reasoning_chains)}条推理链进行重排")
        
        # 将假设输出嵌入向量化
        hypothesis_embedding = self.embeddings.embed_query(hypothesis_output)
        
        # 计算每个推理链的相关性分数
        scored_chains = []
        for chain in reasoning_chains:
            # 将推理链转化为文本表示
            chain_text = self._chain_to_text(chain)
            chain_embedding = self.embeddings.embed_query(chain_text)
            
            # 计算余弦相似度
            similarity = np.dot(hypothesis_embedding, chain_embedding) / (
                np.linalg.norm(hypothesis_embedding) * np.linalg.norm(chain_embedding)
            )
            
            scored_chains.append((chain, similarity))
            
        # 按相似度排序并选取前top_k个链
        ranked_chains = sorted(scored_chains, key=lambda x: x[1], reverse=True)
        top_chains = [chain for chain, score in ranked_chains[:self.top_k]]
        
        logger.info(f"重排后保留了{len(top_chains)}条推理链")
        return top_chains
    
    def _chain_to_text(self, chain: Dict) -> str:
        """将推理链转换为文本表示"""
        text_parts = []
        
        for i, node in enumerate(chain["nodes"]):
            node_text = node.get("name", node.get("id", ""))
            if node_text:
                text_parts.append(node_text)
            
            if i < len(chain["nodes"]) - 1 and i < len(chain["relationships"]):
                text_parts.append(f"--[{chain['relationships'][i]['type']}]-->")
                
        return " ".join(text_parts)
    
    def organize_knowledge(self, query: str, reasoning_chains: List[Dict]) -> str:
        """组织检索到的知识和用户查询为提示（算法步骤9）
        
        Args:
            query: 用户查询
            reasoning_chains: 推理链列表
            
        Returns:
            组织好的提示
        """
        knowledge_text = ""
        
        for i, chain in enumerate(reasoning_chains, 1):
            knowledge_text += f"推理链 {i}:\n"
            knowledge_text += self._chain_to_text(chain) + "\n\n"
        
        prompt = f"""
        用户查询: {query}
        
        以下是与查询相关的知识图谱信息:
        {knowledge_text}
        
        请根据上述知识，准确回答用户的查询。如果知识图谱中的信息不足以回答查询，请说明并提供可能的解答方向。
        """
        
        return prompt
    
    def rag_process(self, query: str) -> Dict[str, Any]:
        """执行完整的HyKGE RAG过程（算法主流程）
        
        Args:
            query: 用户查询
            
        Returns:
            包含增强回答和中间结果的字典
        """
        logger.info(f"开始HyKGE RAG过程，用户查询: {query}")
        
        # 步骤4: 获取假设输出
        hypothesis_prompt = f"请回答以下问题: {query}"
        hypothesis_output = self.llm.invoke(hypothesis_prompt)
        logger.info(f"获取假设输出完成: {hypothesis_output[:100]}...")
        
        # 步骤5: 从假设输出和查询中提取实体
        query_entities = self.extract_entities(query)
        ho_entities = self.extract_entities(hypothesis_output)
        all_entities = query_entities + ho_entities
        
        # 步骤6: 实体链接
        linked_entity_ids = self.entity_linking(all_entities)
        
        # 如果没有找到实体，使用向量搜索回退
        if not linked_entity_ids:
            logger.warning("未找到链接实体，回退到向量搜索")
            documents = self.graph_retriever.vector_search(query, limit=3)
            docs_content = [doc.page_content for doc in documents] if documents else []
            
            final_answer = self.llm.invoke(f"""
            用户查询: {query}
            
            根据以下信息回答用户查询:
            {docs_content}
            """)
            
            return {
                "answer": final_answer,
                "knowledge": docs_content,
                "method": "vector_fallback",
                "entities": all_entities
            }
        
        # 步骤7: 检索推理链
        reasoning_chains = self.retrieve_reasoning_chains(linked_entity_ids)
        
        # 如果没有推理链，回退到直接实体关系获取
        if not reasoning_chains:
            logger.warning("未找到推理链，回退到直接实体关系获取")
            entity_relations = []
            for entity_id in linked_entity_ids:
                relations = self.graph_retriever.get_entity_relations(entity_id)
                if relations:
                    entity_relations.append(relations)
            
            if not entity_relations:
                # 如果仍然没有找到相关信息，回退到向量搜索
                documents = self.graph_retriever.vector_search(query, limit=3)
                docs_content = [doc.page_content for doc in documents] if documents else []
                
                final_answer = self.llm.invoke(f"""
                用户查询: {query}
                
                根据以下信息回答用户查询:
                {docs_content}
                """)
                
                return {
                    "answer": final_answer,
                    "knowledge": docs_content,
                    "method": "vector_fallback",
                    "entities": all_entities,
                    "linked_entities": linked_entity_ids
                }
            
            # 整合实体关系信息
            relations_text = json.dumps(entity_relations, ensure_ascii=False, indent=2)
            final_prompt = f"""
            用户查询: {query}
            
            以下是与查询相关的实体关系信息:
            {relations_text}

            请根据上述信息，准确回答用户的查询。如果知识图谱中的信息不足以回答查询，请说明并提供可能的解答方向。
            """

            final_answer = self.llm.invoke(final_prompt)
            
            return {
                "answer": final_answer,
                "knowledge": entity_relations,
                "method": "entity_relations",
                "entities": all_entities,
                "linked_entities": linked_entity_ids
            }
        
        # 步骤8: 重排和过滤推理链
        pruned_chains = self.rerank_reasoning_chains(reasoning_chains, hypothesis_output)
        
        # 步骤9-10: 组织知识并获取优化的回答
        final_prompt = self.organize_knowledge(query, pruned_chains)
        logger.info(f"Final prompt to LLM: {final_prompt}")
        final_answer = self.llm.invoke(final_prompt)

        return {
            "answer": final_answer,
            "hypothesis_output": hypothesis_output,
            "entities": all_entities,
            "linked_entities": linked_entity_ids,
            "reasoning_chains": pruned_chains,
            "knowledge": [self._chain_to_text(chain) for chain in pruned_chains],
            "method": "hykge"
        }