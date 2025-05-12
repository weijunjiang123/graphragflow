"""
GraphRAG API客户端示例
演示如何调用text2Cypher API服务
"""
import requests
import json
import argparse

def call_text2cypher_api(query, limit=5, base_url="http://localhost:8000"):
    """
    调用text2Cypher API
    
    Args:
        query: 自然语言查询
        limit: 结果数量限制
        base_url: API服务基础URL
        
    Returns:
        API响应
    """
    url = f"{base_url}/api/text2cypher"
    payload = {
        "query": query,
        "limit": limit
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # 检查响应状态
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API调用失败: {e}")
        return None

def call_search_api(query, search_type="text2cypher", limit=5, base_url="http://localhost:8000"):
    """
    调用统一搜索API
    
    Args:
        query: 搜索查询
        search_type: 搜索类型 (text2cypher, hybrid, vector, fulltext)
        limit: 结果数量限制
        base_url: API服务基础URL
        
    Returns:
        API响应
    """
    url = f"{base_url}/api/search"
    payload = {
        "query": query,
        "search_type": search_type,
        "limit": limit
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # 检查响应状态
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API调用失败: {e}")
        return None

def get_examples(base_url="http://localhost:8000"):
    """
    获取示例查询
    
    Args:
        base_url: API服务基础URL
        
    Returns:
        示例查询列表
    """
    url = f"{base_url}/api/examples"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # 检查响应状态
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"获取示例失败: {e}")
        return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GraphRAG API客户端示例")
    parser.add_argument("--query", help="要执行的查询")
    parser.add_argument("--type", default="text2cypher", choices=["text2cypher", "hybrid", "vector", "fulltext"], help="搜索类型")
    parser.add_argument("--limit", type=int, default=5, help="结果数量限制")
    parser.add_argument("--url", default="http://localhost:8000", help="API服务URL")
    parser.add_argument("--examples", action="store_true", help="获取示例查询")
    
    args = parser.parse_args()
    
    # 获取示例查询
    if args.examples:
        examples = get_examples(args.url)
        if examples:
            print("\n示例查询:")
            for i, example in enumerate(examples):
                print(f"{i+1}. {example['description']}")
                print(f"   查询: {example['query']}")
                print(f"   说明: {example['explanation']}")
                print()
        return
    
    # 执行查询
    if not args.query:
        query = input("请输入自然语言查询: ")
    else:
        query = args.query
    
    print(f"\n执行{args.type}搜索: '{query}'\n")
    
    if args.type == "text2cypher":
        result = call_text2cypher_api(query, args.limit, args.url)
    else:
        result = call_search_api(query, args.type, args.limit, args.url)
    
    if result:
        # 打印查询信息
        print(f"原始查询: {result.get('original_query')}")
        
        if args.type == "text2cypher":
            print(f"Cypher查询: {result.get('cypher_query')}")
        
        if result.get('explanation'):
            print(f"\n结果解释: {result.get('explanation')}")
        
        # 打印结果摘要
        print(f"\n结果数量: {result.get('result_count', 0)}")
        
        # 打印结果
        results = result.get('results', [])
        if results:
            print("\n部分结果:")
            for i, record in enumerate(results[:3]):  # 只显示前3条结果
                print(f"结果 {i+1}:")
                print(json.dumps(record, indent=2, ensure_ascii=False))
                print()

if __name__ == "__main__":
    main()
