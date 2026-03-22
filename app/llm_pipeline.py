from rag_retriever import retrieve
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_optimized_content(user_query):
    # 1. Retrieve context
    results = retrieve(user_query, top_k=3)

    context = "\n\n".join([r["text"] for r in results])

    # 2. Build prompt
    prompt = f"""
You are an expert SEO and health content writer for a supplement company.

Using the context below, generate a high-quality product description.

Rules:
- Improve clarity and SEO
- Keep claims realistic and safe (no medical exaggeration)
- Use only information from the context
- Make it engaging and customer-friendly

Context:
{context}

User request:
{user_query}

Output:
"""

    # 3. Call LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You generate safe, SEO-optimised supplement content."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    output = generate_optimized_content("melatonin supplement for sleep")
    print(output)