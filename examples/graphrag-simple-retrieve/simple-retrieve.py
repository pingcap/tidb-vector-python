from sqlalchemy import create_engine, text
import openai
import getpass

# TiDB Connection String Pattern:
# mysql+pymysql://{TIDB_USER}:{TIDB_PASSWORD}@{TIDB_HOST}:{TIDB_PORT}/{TIDB_DB_NAME}?ssl_verify_cert=True&ssl_verify_identity=True

db_engine = create_engine(getpass.getpass("Input your TIDB connection string:"))
oai_cli = openai.OpenAI(api_key=getpass.getpass("Input your OpenAI API Key:"))
question = input("Enter your question:")
embedding = str(oai_cli.embeddings.create(input=[question], model="text-embedding-3-small").data[0].embedding)

with db_engine.connect() as conn:
    result = conn.execute(text("""
    WITH initial_entity AS (
        SELECT id FROM `entities`
        ORDER BY VEC_Cosine_Distance(description_vec, :embedding) LIMIT 1
    ), entities_ids AS (
        SELECT source_entity_id i FROM relationships r INNER JOIN initial_entity i ON r.target_entity_id = i.id
        UNION SELECT target_entity_id i FROM relationships r INNER JOIN initial_entity i ON r.source_entity_id = i.id
        UNION SELECT initial_entity.id i FROM initial_entity
    ) SELECT description FROM `entities` WHERE id IN (SELECT i FROM entities_ids);"""), {"embedding": embedding}).fetchall()

    print(oai_cli.chat.completions.create(model="gpt-4o", messages=[
        {"role": "system", "content": f"Please carefully answer the question by {str(result)}"},
        {"role": "user", "content": question}]).choices[0].message.content)
