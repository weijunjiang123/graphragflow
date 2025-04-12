from neo4j import GraphDatabase

uri = "neo4j://localhost:7687"  # 更换为您的 URI
username = "neo4j"  # 更换为您的用户名
password = "your_password"  # 更换为您的密码

try:
    driver = GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()
    print("连接成功!")
    driver.close()
except Exception as e:
    print(f"连接失败: {e}")